from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional
import uvicorn

from behaviormonitor import BehaviorBackend
from frauddetect import FraudDetection
from frontend_requirements import UserAuthData, BehaviorMetrics, FraudMetrics

# Initialize FastAPI app
app = FastAPI(title="Behavior Monitoring & Fraud Detection API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
SECRET_KEY = "your-secret-key"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Session storage
active_sessions: Dict[str, Dict] = {}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate JWT token and return current user"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return user_id

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/api/auth/login")
async def login(auth_data: UserAuthData):
    """Handle user login and start behavior monitoring"""
    # In production, validate against your user database
    # For demo, we'll accept any username/password combo
    try:
        # Create session
        session_id = f"session_{auth_data.username}_{int(datetime.now().timestamp())}"
        active_sessions[session_id] = {
            "user_id": auth_data.username,
            "start_time": datetime.now(),
            "device_info": auth_data.profile_data.get("device_info", {})
        }
        
        # Create access token
        access_token = create_access_token({"sub": auth_data.username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/session/start")
async def start_session(current_user: str = Depends(get_current_user)):
    """Start a new monitoring session"""
    session_id = f"session_{current_user}_{int(datetime.now().timestamp())}"
    active_sessions[session_id] = {
        "user_id": current_user,
        "start_time": datetime.now(),
        "behavior_metrics": [],
        "fraud_metrics": None
    }
    return {"sessionId": session_id}

@app.post("/api/behavior/typing")
async def record_typing(
    metrics: Dict,
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Record typing behavior metrics"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session["behavior_metrics"].append({
        "type": "typing",
        "data": metrics,
        "timestamp": datetime.now()
    })
    
    # Update behavior monitor
    await behavior_backend.update_typing_metrics(current_user, metrics)
    
    # Check for suspicious behavior
    risk_score = fraud_detector.analyze_typing_pattern(metrics)
    if risk_score > fraud_detector.detection_threshold:
        return {"warning": "Suspicious typing pattern detected"}
    
    return {"status": "recorded"}

@app.post("/api/behavior/mouse")
async def record_mouse(
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
async def record_device_info(
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

@app.post("/api/fraud/check")
async def check_fraud(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Check for fraudulent behavior"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    behavior_metrics = behavior_backend.get_session_metrics(session_id)
    
    # Calculate risk score based on behavior metrics
    risk_score = behavior_metrics.get('risk_score', 0.0)
    anomaly_score = behavior_metrics.get('anomaly_score', 0.0)
    
    warnings = []
    if risk_score > 0.7:
        warnings.append("High risk behavior detected")
    if anomaly_score > 0.8:
        warnings.append("Significant behavioral anomalies detected")
    
    return {
        "risk_score": risk_score,
        "anomaly_score": anomaly_score,
        "warnings": warnings
    }

@app.get("/api/session/status")
async def get_session_status(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get session status"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    behavior_metrics = behavior_backend.get_session_metrics(session_id)
    
    return {
        "duration": behavior_metrics.get('duration', 0),
        "typing_speed": behavior_metrics.get('typing_speed', 0),
        "risk_score": behavior_metrics.get('risk_score', 0),
        "anomaly_score": behavior_metrics.get('anomaly_score', 0)
    }

@app.get("/")
async def read_root(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 