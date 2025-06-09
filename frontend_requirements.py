from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel, EmailStr, validator, constr
import re

class UserAuthData(BaseModel):
    """User authentication and account data"""
    username: constr(min_length=3, max_length=50)
    password: str  # Should be hashed before storage
    email: EmailStr
    account_creation_date: datetime
    last_login: datetime
    account_status: str  # active/locked/suspended
    profile_data: Dict[str, Any]  # Additional user profile information
    
    @validator('password')
    def password_must_be_strong(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError("Password must contain at least one special character")
        return v
    
    @validator('account_status')
    def validate_account_status(cls, v):
        """Validate account status"""
        valid_statuses = ['active', 'locked', 'suspended']
        if v.lower() not in valid_statuses:
            raise ValueError(f"Account status must be one of: {', '.join(valid_statuses)}")
        return v.lower()
    
    @validator('profile_data')
    def validate_profile_data(cls, v):
        """Validate profile data structure"""
        required_fields = ['device_info', 'location', 'preferences']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Profile data must contain {field}")
        return v

class DeviceInfo(BaseModel):
    """Device information model"""
    device_id: str
    device_type: str
    os_name: str
    os_version: str
    browser_name: str
    browser_version: str
    screen_width: int
    screen_height: int
    is_mobile: bool
    is_tablet: bool
    
    @validator('screen_width', 'screen_height')
    def validate_screen_dimensions(cls, v):
        """Validate screen dimensions"""
        if v <= 0 or v > 10000:  # Unrealistic screen dimensions
            raise ValueError("Invalid screen dimension")
        return v

class Location(BaseModel):
    """Location information model"""
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    ip_address: str
    
    @validator('latitude')
    def validate_latitude(cls, v):
        """Validate latitude"""
        if not -90 <= v <= 90:
            raise ValueError("Invalid latitude")
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        """Validate longitude"""
        if not -180 <= v <= 180:
            raise ValueError("Invalid longitude")
        return v
    
    @validator('ip_address')
    def validate_ip(cls, v):
        """Validate IP address format"""
        ipv4_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        ipv6_pattern = re.compile(r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')
        if not ipv4_pattern.match(v) and not ipv6_pattern.match(v):
            raise ValueError("Invalid IP address format")
        return v

class BehaviorMetrics(BaseModel):
    """Real-time behavior monitoring metrics"""
    # Session identification
    user_id: str
    session_id: str
    timestamp: datetime
    
    # Typing behavior
    typing_speed: float  # characters per second
    key_press_duration: List[float]  # milliseconds for each key press
    key_intervals: List[float]  # time between key presses
    backspace_frequency: float
    special_key_usage: Dict[str, int]  # frequency of special key usage
    
    # Mouse/Pointer behavior
    mouse_movement_pattern: List[Dict[str, float]]  # x, y coordinates and speed
    click_pattern: List[Dict[str, Any]]  # location, duration, type
    scroll_pattern: List[Dict[str, float]]  # speed, direction
    
    # Device/Environment data
    device_info: DeviceInfo
    location: Location
    
    # Page interaction
    page_focus_time: float
    idle_time: float
    form_fill_pattern: Dict[str, float]  # field name -> time taken
    
    @validator('typing_speed')
    def validate_typing_speed(cls, v):
        """Validate typing speed"""
        if v < 0 or v > 1000:  # Unrealistic typing speeds
            raise ValueError("Invalid typing speed")
        return v
    
    @validator('key_press_duration', 'key_intervals')
    def validate_timing_lists(cls, v):
        """Validate timing measurements"""
        if not v:
            raise ValueError("Timing list cannot be empty")
        if any(t < 0 or t > 5000 for t in v):  # Unrealistic timings
            raise ValueError("Invalid timing value")
        return v
    
    @validator('backspace_frequency')
    def validate_backspace_freq(cls, v):
        """Validate backspace frequency"""
        if not 0 <= v <= 1:
            raise ValueError("Backspace frequency must be between 0 and 1")
        return v
    
    @validator('page_focus_time', 'idle_time')
    def validate_time_measurements(cls, v):
        """Validate time measurements"""
        if v < 0:
            raise ValueError("Time measurements cannot be negative")
        return v

class FraudMetrics(BaseModel):
    """Fraud detection and risk assessment metrics"""
    # Risk scores
    risk_score: float
    anomaly_score: float
    confidence_score: float
    
    # Session security
    session_duration: int
    authentication_method: str
    mfa_status: bool
    session_continuity: float  # measure of session consistency
    
    # Activity metrics
    transaction_count: int
    high_risk_actions: List[Dict[str, Any]]
    failed_attempts: int
    password_change_history: List[datetime]
    
    # Location/Device security
    known_devices: List[str]
    trusted_locations: List[Location]
    vpn_detection: bool
    proxy_detection: bool
    
    # Historical patterns
    typical_activity_hours: List[int]
    typical_locations: List[Location]
    typical_devices: List[str]
    
    @validator('risk_score', 'anomaly_score', 'confidence_score', 'session_continuity')
    def validate_scores(cls, v):
        """Validate score values"""
        if not 0 <= v <= 1:
            raise ValueError("Score values must be between 0 and 1")
        return v
    
    @validator('session_duration', 'transaction_count', 'failed_attempts')
    def validate_counts(cls, v):
        """Validate count values"""
        if v < 0:
            raise ValueError("Count values cannot be negative")
        return v
    
    @validator('typical_activity_hours')
    def validate_hours(cls, v):
        """Validate hour values"""
        if not all(0 <= h <= 23 for h in v):
            raise ValueError("Hour values must be between 0 and 23")
        return v

# Frontend data collection functions
class FrontendDataCollector:
    """Collects and processes frontend behavioral data"""
    
    def __init__(self):
        self.current_session: Optional[str] = None
        self.behavior_metrics: List[BehaviorMetrics] = []
        self.fraud_metrics: Optional[FraudMetrics] = None
    
    async def start_session(self, user_id: str) -> str:
        """Initialize a new monitoring session"""
        if not user_id:
            raise ValueError("User ID cannot be empty")
        self.current_session = f"session_{user_id}_{int(datetime.now().timestamp())}"
        return self.current_session
    
    async def collect_typing_metrics(self, event_data: Dict[str, Any]) -> None:
        """Collect typing behavior metrics from frontend events"""
        if not self.current_session:
            raise ValueError("No active session")
        if not event_data:
            raise ValueError("Event data cannot be empty")
        # Implementation for collecting typing metrics
        pass
    
    async def collect_mouse_metrics(self, event_data: Dict[str, Any]) -> None:
        """Collect mouse/pointer behavior metrics"""
        if not self.current_session:
            raise ValueError("No active session")
        if not event_data:
            raise ValueError("Event data cannot be empty")
        # Implementation for collecting mouse metrics
        pass
    
    async def collect_device_info(self) -> Dict[str, Any]:
        """Collect device and environment information"""
        if not self.current_session:
            raise ValueError("No active session")
        # Implementation for collecting device info
        pass
    
    async def analyze_behavior(self) -> Tuple[BehaviorMetrics, FraudMetrics]:
        """Analyze collected behavior data and generate metrics"""
        if not self.current_session:
            raise ValueError("No active session")
        if not self.behavior_metrics:
            raise ValueError("No behavior metrics collected")
        # Implementation for behavior analysis
        pass

# Required frontend JavaScript functions
REQUIRED_JS_FUNCTIONS = """
// Typing behavior monitoring
function monitorKeyPress(event) {
    const metrics = {
        key: event.key,
        timestamp: Date.now(),
        duration: event.duration,
        target: event.target.id
    };
    sendToBackend('/api/behavior/typing', metrics);
}

// Mouse behavior monitoring
function monitorMouseMovement(event) {
    const metrics = {
        x: event.clientX,
        y: event.clientY,
        timestamp: Date.now(),
        target: event.target.id
    };
    sendToBackend('/api/behavior/mouse', metrics);
}

// Device information collection
function collectDeviceInfo() {
    const info = {
        userAgent: navigator.userAgent,
        screen: {
            width: screen.width,
            height: screen.height
        },
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        language: navigator.language
    };
    sendToBackend('/api/behavior/device', info);
}

// Session management
function startMonitoringSession() {
    // Initialize session
    const sessionId = await initSession();
    
    // Attach event listeners
    document.addEventListener('keypress', monitorKeyPress);
    document.addEventListener('mousemove', monitorMouseMovement);
    
    // Collect initial device info
    collectDeviceInfo();
    
    // Start periodic data collection
    setInterval(collectBehaviorMetrics, 5000);
}
"""

# Required API endpoints
REQUIRED_ENDPOINTS = [
    {
        "path": "/api/auth/login",
        "method": "POST",
        "data": UserAuthData
    },
    {
        "path": "/api/behavior/typing",
        "method": "POST",
        "data": Dict[str, Any]  # Typing metrics
    },
    {
        "path": "/api/behavior/mouse",
        "method": "POST",
        "data": Dict[str, Any]  # Mouse metrics
    },
    {
        "path": "/api/behavior/device",
        "method": "POST",
        "data": Dict[str, Any]  # Device info
    },
    {
        "path": "/api/fraud/check",
        "method": "POST",
        "data": FraudMetrics
    }
] 