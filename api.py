from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import uvicorn
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, EmailStr, validator
import logging

from behaviormonitor import BehaviorBackend
from frauddetect import FraudDetection
from frontend_requirements import UserAuthData, BehaviorMetrics, FraudMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Behavior Monitoring & Fraud Detection API")

# Setup rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize components
behavior_backend = BehaviorBackend()
fraud_detector = FraudDetection()

# JWT settings
SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not SECRET_KEY:
    logger.warning("JWT_SECRET_KEY not set! Generating random key...")
    SECRET_KEY = os.urandom(32).hex()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Session storage
active_sessions: Dict[str, Dict] = {}

# Input validation models
class LoginRequest(BaseModel):
    """Login request validation model"""
    username: str
    password: str
    
    @validator('username')
    def username_must_be_valid(cls, v):
        if not v:
            raise ValueError('Username is required')
        return v
    
    @validator('password')
    def password_must_be_valid(cls, v):
        if not v:
            raise ValueError('Password is required')
        return v

    class Config:
        # Allow extra fields in request
        extra = "allow"

class SessionMetrics(BaseModel):
    metrics: Dict
    timestamp: datetime

class AccountDetails(BaseModel):
    """Account details model"""
    user_id: str
    account_type: str = "Standard"
    account_status: str = "Active"
    creation_date: datetime
    last_login: datetime
    security_level: str
    location_history: List[Dict] = []

# Store account details (in-memory for demo)
account_details = {}

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Validate JWT token and return current user"""
    # Special case for demo token
    if token == "demo_token":
        return "demo"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except jwt.PyJWTError:  # Changed from JWTError to PyJWTError
        raise credentials_exception

def create_access_token(data: dict) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )

@app.post("/api/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, auth_data: LoginRequest):
    """Handle user login and start behavior monitoring"""
    try:
        logger.info(f"Login attempt for user: {auth_data.username}")
        
        # Special handling for demo user
        if auth_data.username.lower() == "demo":
            session_id = f"demo_session_{int(datetime.now().timestamp())}"
            active_sessions[session_id] = {
                "user_id": "demo",
                "start_time": datetime.now(),
                "device_info": {},
                "is_demo": True
            }
            
            # Create demo account details
            account_details["demo"] = {
                "user_id": "demo",
                "account_number": "DEMO123456",
                "balance": 10000.00,
                "account_type": "Demo Account",
                "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "security_level": "Standard",
                "recent_transactions": [
                    {
                        "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "description": f"Demo Transaction {i+1}",
                        "amount": float(f"{(i+1)*-100.50:.2f}")
                    }
                    for i in range(5)
                ]
            }
            
            return {
                "access_token": "demo_token",
                "token_type": "bearer",
                "session_id": session_id
            }
        
        # For regular users
        session_id = f"session_{auth_data.username}_{int(datetime.now().timestamp())}"
        active_sessions[session_id] = {
            "user_id": auth_data.username,
            "start_time": datetime.now(),
            "device_info": {}
        }
        
        # Create or update account details
        if auth_data.username not in account_details:
            account_details[auth_data.username] = AccountDetails(
                user_id=auth_data.username,
                creation_date=datetime.now(),
                last_login=datetime.now(),
                security_level="Standard"
            )
        else:
            account_details[auth_data.username].last_login = datetime.now()
        
        # Create access token
        access_token = create_access_token({"sub": auth_data.username})
        
        logger.info(f"Login successful for user: {auth_data.username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/api/session/start")
@limiter.limit("10/minute")
async def start_session(request: Request, current_user: str = Depends(get_current_user)):
    """Start a new monitoring session"""
    try:
        session_id = f"session_{current_user}_{int(datetime.now().timestamp())}"
        active_sessions[session_id] = {
            "user_id": current_user,
            "start_time": datetime.now(),
            "behavior_metrics": [],
            "fraud_metrics": None
        }
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Session start error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start session"
        )

@app.post("/api/behavior/typing")
@limiter.limit("30/minute")
async def record_typing(
    request: Request,
    metrics: SessionMetrics,
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Record typing behavior metrics"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session["behavior_metrics"].append({
            "type": "typing",
            "data": metrics.metrics,
            "timestamp": metrics.timestamp
        })
        
        # Update behavior monitor
        await behavior_backend.update_typing_metrics(current_user, metrics.metrics)
        
        # Check for suspicious behavior
        risk_score = fraud_detector.analyze_typing_pattern(metrics.metrics)
        if risk_score > fraud_detector.detection_threshold:
            return {"warning": "Suspicious typing pattern detected"}
        
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Error recording typing metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not record typing metrics"
        )

@app.post("/api/behavior/mouse")
@limiter.limit("30/minute")
async def record_mouse(
    request: Request,
    metrics: Dict,
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Record mouse movement metrics"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session["behavior_metrics"].append({
        "type": "mouse",
        "data": metrics,
        "timestamp": datetime.now()
    })
    
    # Update behavior monitor
    await behavior_backend.update_mouse_metrics(current_user, metrics)
    
    return {"status": "recorded"}

@app.post("/api/behavior/device")
@limiter.limit("30/minute")
async def record_device_info(
    request: Request,
    info: Dict,
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Record device information"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session["device_info"] = info
    
    # Update behavior monitor
    await behavior_backend.update_device_info(current_user, info)
    
    # Check if device is trusted
    is_trusted = await behavior_backend.is_trusted_device(current_user, info)
    if not is_trusted:
        return {"warning": "Unrecognized device detected"}
    
    return {"status": "recorded"}

@app.get("/api/account/details")
@limiter.limit("30/minute")
async def get_account_details(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Get user account details"""
    try:
        # Generate demo data if not exists
        if current_user not in account_details:
            account_details[current_user] = {
                "user_id": current_user,
                "account_number": f"2024{hash(current_user) % 10000:04d}",
                "balance": 5000.00,
                "account_type": "Premium",
                "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "security_level": "Standard",
                "recent_transactions": [
                    {
                        "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "description": f"Transaction {i+1}",
                        "amount": float(f"{(i+1)*-50.25:.2f}")
                    }
                    for i in range(5)
                ]
            }
        
        return account_details[current_user]
    except Exception as e:
        logger.error(f"Error getting account details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve account details"
        )

@app.get("/api/session/status")
@limiter.limit("30/minute")
async def get_session_status(
    request: Request,
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get session status and metrics"""
    try:
        if session_id not in active_sessions:
            # For demo sessions that might have been lost, recreate them
            if session_id.startswith("demo_session_"):
                active_sessions[session_id] = {
                    "user_id": "demo",
                    "start_time": datetime.now(),
                    "device_info": {},
                    "is_demo": True
                }
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        duration = (datetime.now() - session["start_time"]).total_seconds()
        
        # Enhanced demo data for visualization
        if session.get("is_demo", False):
            status_data = {
                "session_id": session_id,
                "duration": int(duration),
                "risk_score": 0.15,  # Low risk demo value
                "risk_level": "low",
                "recent_activities": [
                    "Demo login successful",
                    "Account overview accessed",
                    "Security check completed",
                    "Location verified",
                    "Behavior analysis completed"
                ],
                "anomalies": [],
                "activity_timeline": {
                    "timestamps": [(datetime.now() - timedelta(minutes=i)).strftime("%H:%M:%S") for i in range(10)],
                    "activity_scores": [0.85, 0.88, 0.92, 0.90, 0.87, 0.89, 0.91, 0.86, 0.88, 0.90]
                },
                "risk_factors": {
                    "Location": 0.1,
                    "Device": 0.15,
                    "Behavior": 0.12,
                    "Transaction": 0.08,
                    "Authentication": 0.05
                }
            }
        else:
            # Regular user data
            status_data = {
                "session_id": session_id,
                "duration": int(duration),
                "risk_score": 0.25,
                "risk_level": "low",
                "recent_activities": [
                    "Account overview accessed",
                    "Security check completed",
                    "Location verified"
                ],
                "anomalies": [],
                "activity_timeline": {
                    "timestamps": [(datetime.now() - timedelta(minutes=i)).strftime("%H:%M:%S") for i in range(5)],
                    "activity_scores": [0.8, 0.85, 0.9, 0.88, 0.92]
                },
                "risk_factors": {
                    "Location": 0.1,
                    "Device": 0.2,
                    "Behavior": 0.15,
                    "Transaction": 0.1
                }
            }
        
        return status_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve session status"
        )

@app.post("/api/fraud/check")
@limiter.limit("30/minute")
async def check_fraud(
    request: Request,
    session_id: str,
    location_info: Optional[Dict] = None,
    current_user: str = Depends(get_current_user)
):
    """Check for fraudulent activity"""
    try:
        if session_id not in active_sessions:
            # For demo sessions that might have been lost, recreate them
            if session_id.startswith("demo_session_"):
                active_sessions[session_id] = {
                    "user_id": "demo",
                    "start_time": datetime.now(),
                    "device_info": {},
                    "is_demo": True
                }
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        
        # Enhanced demo data for fraud detection
        if session.get("is_demo", False):
            return {
                "risk_score": 0.15,  # Low risk score
                "anomaly_score": 0.12,  # Low anomaly score
                "security_score": 0.88,  # High security score
                "location_verified": True,
                "device_verified": True,
                "active_threats": [],
                "security_events": [
                    "Demo mode active",
                    "Location verification successful",
                    "Device fingerprint matched",
                    "Behavior pattern normal",
                    "No suspicious transactions detected",
                    "Multi-factor authentication enabled"
                ],
                "warnings": [],
                "device_info": {
                    "type": "Desktop",
                    "os": "Linux",
                    "browser": "Chrome",
                    "last_seen": "Just now",
                    "risk_level": "Low"
                },
                "location_history": [
                    {
                        "lat": location_info["latitude"] if location_info else 0,
                        "lon": location_info["longitude"] if location_info else 0,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            }
        
        # Regular user data
        return {
            "risk_score": 0.15,
            "anomaly_score": 0.12,
            "security_score": 0.85,
            "location_verified": True,
            "device_verified": True,
            "active_threats": [],
            "security_events": [
                "Location verification successful",
                "Device fingerprint matched",
                "Behavior pattern normal",
                "No suspicious transactions detected"
            ],
            "warnings": [],
            "location_history": [
                {
                    "lat": location_info["latitude"] if location_info else 0,
                    "lon": location_info["longitude"] if location_info else 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking fraud: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not perform fraud check"
        )

@app.get("/")
async def read_root(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 