import streamlit as st
from streamlit_javascript import st_javascript
# Page configurations must be the first Streamlit command
st.set_page_config(
    page_title="SecureBank - Behavior Monitoring",
    page_icon="🏦",
    layout="wide"
)

import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin
from typing import Optional, Dict
import time
import json
import pandas as pd
import numpy as np

# Mock data functions for standalone deployment
@st.cache_data
def get_mock_account_details(user_id: str) -> Dict:
    """Generate mock account details"""
    return {
        "user_id": user_id,
        "account_number": f"DEMO{hash(user_id) % 100000:05d}",
        "balance": 10000.00 if user_id == "demo" else 5000.00,
        "account_type": "Demo Account" if user_id == "demo" else "Premium",
        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "creation_date": datetime.now().strftime("%Y-%m-%d"),
        "security_level": "Standard",
        "recent_transactions": [
            {
                "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "description": f"Transaction {i+1}",
                "amount": float(f"{(i+1)*-100.50:.2f}")
            }
            for i in range(5)
        ]
    }

@st.cache_data
def get_mock_session_status(session_id: str) -> Dict:
    """Generate mock session status"""
    return {
        "session_id": session_id,
        "duration": int(time.time() % 3600),  # Mock duration
        "risk_score": 0.15,
        "security_score": 0.85,
        "anomaly_score": 0.12,
        "activity_timeline": [
            {"time": f"{i:02d}:00", "activity": np.random.randint(10, 100)}
            for i in range(24)
        ],
        "warnings": []
    }

@st.cache_data
def get_mock_fraud_check(session_id: str) -> Dict:
    """Generate mock fraud check results"""
    return {
        "risk_score": 0.15,
        "security_score": 0.85,
        "anomaly_score": 0.12,
        "warnings": [],
        "risk_factors": {
            "location_risk": 0.1,
            "device_risk": 0.05,
            "behavior_risk": 0.08,
            "time_risk": 0.02
        }
    }

# Constants - keeping for compatibility but will be overridden
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffbb33;
        font-weight: bold;
    }
    .risk-low {
        color: #00C851;
        font-weight: bold;
    }
    .banking-header {
        background-color: #1E88E5;
        padding: 1rem;
        color: white;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_client_location():
    """Get client's location using JS to get IP, then fetch geolocation"""
    try:
        # Step 1: Get IP address from client browser
        ip_data = st_javascript("""await fetch('https://api.ipify.org?format=json')
                                  .then(res => res.json())""")
        if not ip_data or 'ip' not in ip_data:
            st.warning("Could not retrieve client IP.")
            return None

        ip = ip_data['ip']

        # Step 2: Use IP to get location info
        response = requests.get(f'https://ipapi.co/{ip}/json/')
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'latitude': data.get('latitude', 0),
                'longitude': data.get('longitude', 0),
                'ip': ip
            }
        else:
            st.warning("Failed to fetch location details.")
    except Exception as e:
        st.error(f"Failed to get location: {str(e)}")
    return None

def make_api_request(endpoint: str, method: str = 'GET', data: Optional[dict] = None,
                    headers: Optional[dict] = None) -> Optional[dict]:
    """Mock API request function for standalone deployment"""
    try:
        # Mock different endpoints with appropriate responses
        if endpoint == "/api/account/details":
            return get_mock_account_details(st.session_state.get('user_id', 'demo'))
        elif endpoint.startswith("/api/session/status"):
            session_id = st.session_state.get('session_id', 'demo_session')
            return get_mock_session_status(session_id)
        elif endpoint.startswith("/api/fraud/check"):
            session_id = st.session_state.get('session_id', 'demo_session')
            return get_mock_fraud_check(session_id)
        elif endpoint == "/api/auth/login":
            # Mock login response
            return {
                "access_token": "demo_token",
                "token_type": "bearer",
                "session_id": f"session_{int(time.time())}"
            }
        else:
            # Default mock response for any other endpoints
            return {"status": "success", "message": "Mock response"}
    except Exception as e:
        # Don't show error for mock requests, just return None
        return None

def login(username: str, password: str) -> bool:
    """Handle user login - accepts any credentials for demo"""
    if not username or not password:
        st.error("Username and password are required")
        return False

    # Accept any username/password for demo purposes
    st.session_state.authenticated = True
    st.session_state.user_id = username
    st.session_state.session_id = f"{username}_session_{int(time.time())}"
    st.session_state.access_token = "demo_token"
    return True

def show_login_page():
    """Display the login page"""
    st.markdown('<div class="banking-header"><h1>🏦 SecureBank</h1></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Welcome to SecureBank")
        st.markdown("Please login to access your account")

        # Demo user information
        st.info("""
        👋 **Demo Mode Active!**
        - Username: `demo` (or any username)
        - Password: `demo` (or any password)

        This demo accepts any credentials and provides access to simulated banking data and security features.
        """)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if login(username, password):
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()

def show_dashboard():
    """Display the main dashboard after login"""
    st.markdown('<div class="banking-header"><h1>🏦 SecureBank Dashboard</h1></div>', unsafe_allow_html=True)

    # Get client location
    location_info = get_client_location()

    # Get account details
    account_details = make_api_request(
        endpoint="/api/account/details"
    )

    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.user_id}")
        if location_info:
            st.markdown("---")
            st.markdown("### 📍 Your Location")
            st.markdown(f"**City:** {location_info['city']}")
            st.markdown(f"**Region:** {location_info['region']}")
            st.markdown(f"**Country:** {location_info['country']}")
        st.markdown("---")

        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()

        if st.button("📊 Security Monitor", use_container_width=True):
            st.session_state.current_page = "security"
            st.rerun()

        if st.button("📝 Activity Log", use_container_width=True):
            st.session_state.current_page = "activity"
            st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def show_account_overview():
    """Display the account overview section"""
    st.markdown("## 🏦 Account Overview")

    # Get account details
    account_details = make_api_request(
        endpoint="/api/account/details"
    )

    if account_details:
        # Account Summary
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### 💳 Account Information")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.write(f"**Account Number:** {account_details.get('account_number', 'N/A')}")
            st.write(f"**Account Type:** {account_details.get('account_type', 'N/A')}")
            st.write(f"**Account Status:** Active")
            st.write(f"**Last Activity:** {account_details.get('last_activity', 'N/A')}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 💰 Balance")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            balance = account_details.get('balance', 0)
            st.markdown(f"<h2>${balance:,.2f}</h2>", unsafe_allow_html=True)
            st.write("Available Balance")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("### 🔒 Security Level")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            security_level = account_details.get('security_level', 'Standard')
            st.write(f"**Level:** {security_level}")
            st.write(f"**Since:** {account_details.get('creation_date', 'N/A')}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("🔄 Transfer Money", use_container_width=True)
        with col2:
            st.button("💸 Pay Bills", use_container_width=True)
        with col3:
            st.button("📊 View Statements", use_container_width=True)
        with col4:
            st.button("⚙️ Account Settings", use_container_width=True)

        # Recent Transactions
        st.markdown("### 📝 Recent Transactions")
        transactions = account_details.get('recent_transactions', [])
        if transactions:
            for tx in transactions:
                with st.container():
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Date:** {tx['date']}")
                    with col2:
                        st.write(f"**Description:** {tx['description']}")
                    with col3:
                        amount = tx['amount']
                        color = "red" if amount < 0 else "green"
                        st.markdown(f"<span style='color: {color}'>${abs(amount):,.2f}</span>",
                                  unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No recent transactions found")

        # Account Analytics
        st.markdown("### 📈 Account Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Monthly Activity")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.line_chart({
                'Activity': [50, 45, 60, 55, 70, 65, 75]
            })
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Spending Categories")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            spending_data = {
                'Shopping': 30,
                'Bills': 25,
                'Food': 20,
                'Transport': 15,
                'Others': 10
            }
            st.bar_chart(spending_data)
            st.markdown("</div>", unsafe_allow_html=True)

def show_security_page():
    """Display the security monitoring page"""
    st.markdown("## 📊 Security Monitor")

    # Get client location for security context
    location_info = get_client_location()

    # Get session status
    session_status = make_api_request(
        endpoint=f"/api/session/status?session_id={st.session_state.session_id}"
    )

    # Get fraud check results with location context
    fraud_check = make_api_request(
        endpoint=f"/api/fraud/check?session_id={st.session_state.session_id}",
        method="POST",
        data={'location_info': location_info} if location_info else None
    )

    if session_status and fraud_check:
        # Security Overview
        st.markdown("### 🛡️ Security Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Session Duration")
            duration = session_status['duration']
            st.markdown(f"<h3>{duration} sec</h3>", unsafe_allow_html=True)
            st.markdown("Active Time")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Risk Level")
            risk_score = fraud_check['risk_score'] * 100
            risk_class = "risk-low" if risk_score < 30 else "risk-medium" if risk_score < 70 else "risk-high"
            st.markdown(f'<h3 class="{risk_class}">{risk_score:.1f}%</h3>', unsafe_allow_html=True)
            st.markdown("Overall Risk")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Security Score")
            security_score = fraud_check['security_score'] * 100
            st.markdown(f"<h3 style='color: {'green' if security_score > 80 else 'orange'}'>{security_score:.1f}%</h3>",
                       unsafe_allow_html=True)
            st.markdown("Protection Level")
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Anomaly Score")
            anomaly_score = fraud_check['anomaly_score'] * 100
            st.markdown(f"<h3 style='color: {'green' if anomaly_score < 20 else 'red'}'>{anomaly_score:.1f}%</h3>",
                       unsafe_allow_html=True)
            st.markdown("Behavior Analysis")
            st.markdown("</div>", unsafe_allow_html=True)

        # Activity Timeline
        st.markdown("### 📈 Activity Timeline")
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        if 'activity_timeline' in session_status:
            timeline_data = {
                'Time': session_status['activity_timeline']['timestamps'],
                'Score': session_status['activity_timeline']['activity_scores']
            }
            st.line_chart(timeline_data)
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk Factors
        st.markdown("### ⚠️ Risk Factors")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Risk Components")
            risk_factors = session_status.get('risk_factors', {})
            for factor, score in risk_factors.items():
                st.markdown(f"**{factor}**")
                st.progress(score)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown("#### Security Events")
            events = fraud_check.get('security_events', [])
            for event in events:
                st.markdown(f"✓ {event}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Location Security
        if location_info:
            st.markdown("### 📍 Location Security")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.markdown("#### Current Location")
                st.markdown(f"**City:** {location_info['city']}")
                st.markdown(f"**Region:** {location_info['region']}")
                st.markdown(f"**Country:** {location_info['country']}")
                st.markdown(f"**IP Address:** {location_info['ip']}")
                verified = fraud_check.get('location_verified', False)
                st.markdown(f"**Status:** {'✅ Verified' if verified else '⚠️ Unverified'}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.markdown("#### Device Information")
                device_verified = fraud_check.get('device_verified', False)
                st.markdown(f"**Device Status:** {'✅ Trusted Device' if device_verified else '⚠️ New Device'}")
                st.markdown("**Browser:** Chrome")
                st.markdown("**Platform:** Linux")
                st.markdown("**Last Seen:** Now")
                st.markdown("</div>", unsafe_allow_html=True)

        # Active Threats
        threats = fraud_check.get('active_threats', [])
        if threats:
            st.markdown("### 🚨 Active Threats")
            st.markdown('<div class="stat-box risk-high">', unsafe_allow_html=True)
            for threat in threats:
                st.error(threat)
            st.markdown("</div>", unsafe_allow_html=True)

        # Security Recommendations
        st.markdown("### 💡 Security Recommendations")
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        recommendations = [
            "Enable two-factor authentication for enhanced security",
            "Review recent account activities regularly",
            "Keep your contact information up to date",
            "Use a strong, unique password"
        ]
        for rec in recommendations:
            st.info(rec)
        st.markdown("</div>", unsafe_allow_html=True)

def show_activity_page():
    """Display the activity log page"""
    st.markdown("## 📝 Activity Log")

    # Activity Summary
    st.markdown("### 📊 Activity Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown("#### Today's Activity")
        st.markdown("**Login Count:** 2")
        st.markdown("**Transactions:** 3")
        st.markdown("**Security Events:** 1")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown("#### Active Sessions")
        st.markdown("**Current Sessions:** 1")
        st.markdown("**Last Login:** Just now")
        st.markdown("**Device:** Chrome/Linux")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown("#### Security Status")
        st.markdown("**Risk Level:** Low")
        st.markdown("**Alerts:** None")
        st.markdown("**2FA Status:** Enabled")
        st.markdown("</div>", unsafe_allow_html=True)

    # Activity Timeline
    st.markdown("### ⏱️ Recent Activities")

    activities = [
        {
            "timestamp": datetime.now(),
            "type": "Login",
            "details": "Successful login from usual device",
            "category": "authentication",
            "icon": "🔐"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=5),
            "type": "Security Check",
            "details": "Behavioral analysis completed",
            "category": "security",
            "icon": "🛡️"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=10),
            "type": "Transaction",
            "details": "Viewed account balance",
            "category": "account",
            "icon": "💰"
        },
        {
            "timestamp": datetime.now() - timedelta(hours=1),
            "type": "Profile Update",
            "details": "Updated contact information",
            "category": "profile",
            "icon": "👤"
        }
    ]

    # Filter options
    st.markdown("#### Filter Activities")
    col1, col2 = st.columns(2)
    with col1:
        selected_categories = st.multiselect(
            "Category",
            ["All", "Authentication", "Security", "Account", "Profile"],
            default="All"
        )

    with col2:
        time_filter = st.selectbox(
            "Time Period",
            ["Last 24 Hours", "Last Week", "Last Month", "All Time"]
        )

    # Activity list with enhanced visualization
    for activity in activities:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            st.markdown(f"### {activity['icon']}")
            st.markdown(f"**{activity['type']}**")

        with col2:
            st.markdown(f"**Details:** {activity['details']}")
            st.markdown(f"**Category:** {activity['category'].title()}")

        with col3:
            st.markdown(f"**Time:**")
            st.markdown(activity['timestamp'].strftime("%H:%M:%S"))

        st.markdown("</div>", unsafe_allow_html=True)

    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📊 Export to CSV",
            data="",  # Add CSV data here
            file_name="activity_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📑 Export to PDF",
            data="",  # Add PDF data here
            file_name="activity_log.pdf",
            mime="application/pdf",
            use_container_width=True
        )

def show_behavior_monitor():
    """Display the behavior monitoring page"""
    st.markdown("## 👥 Behavior Monitoring")

    # Get session status
    session_id = st.session_state.get('session_id')
    user_id = st.session_state.get('user_id')

    # Demo user data
    demo_user = user_id == "demo"

    if session_id:
        session_status = make_api_request(
            endpoint=f"/api/session/status?session_id={session_id}"
        )

        if session_status:
            if demo_user:
                st.info("🔍 Monitoring demo user behavior patterns")

                # User Selection for Demo
                selected_user = st.selectbox(
                    "Select User Profile",
                    ["John Smith (Regular)", "Alice Johnson (High Risk)", "Bob Wilson (Low Risk)"],
                    index=0
                )

                # Behavioral Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Average Typing Speed", "65 WPM", "↑ 5 WPM")
                    st.metric("Mouse Movement Pattern", "92% Match", "↑ 3%")

                with col2:
                    st.metric("Session Duration", "45 minutes", "↓ 10 min")
                    st.metric("Interaction Consistency", "88%", "↓ 2%")

                with col3:
                    st.metric("Device Trust Score", "95%", "↑ 1%")
                    st.metric("Location Consistency", "High", "")

                # Activity Timeline
                st.subheader("🕒 Activity Timeline")
                activity_data = {
                    'time': pd.date_range(end=datetime.now(), periods=10, freq='1h'),
                    'activity': [
                        "Login from usual device",
                        "Viewed account summary",
                        "Updated profile settings",
                        "Checked transaction history",
                        "Initiated fund transfer",
                        "Reviewed security settings",
                        "Downloaded statement",
                        "Changed notification preferences",
                        "Viewed investment portfolio",
                        "Session ended"
                    ],
                    'risk_level': [
                        "low", "low", "low", "medium", "medium",
                        "low", "low", "low", "low", "low"
                    ]
                }
                df_activity = pd.DataFrame(activity_data)

                for idx, row in df_activity.iterrows():
                    color = "green" if row['risk_level'] == "low" else "orange"
                    st.markdown(
                        f"**{row['time'].strftime('%H:%M')}** - {row['activity']} "
                        f"<span style='color: {color}'>[{row['risk_level'].upper()}]</span>",
                        unsafe_allow_html=True
                    )

                # Behavioral Analysis Charts
                st.subheader("📊 Behavioral Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Typing Pattern Analysis
                    typing_data = {
                        'metric': ['Speed', 'Rhythm', 'Pressure', 'Error Rate', 'Consistency'],
                        'score': [85, 92, 78, 95, 88]
                    }
                    df_typing = pd.DataFrame(typing_data)
                    st.bar_chart(df_typing.set_index('metric'))
                    st.caption("Typing Pattern Analysis")

                with col2:
                    # Mouse Movement Analysis
                    mouse_data = {
                        'time': range(10),
                        'speed': [65, 70, 68, 72, 69, 71, 67, 70, 73, 71]
                    }
                    df_mouse = pd.DataFrame(mouse_data)
                    st.line_chart(df_mouse.set_index('time'))
                    st.caption("Mouse Movement Patterns")

                # Device Usage Statistics
                st.subheader("💻 Device Usage")
                device_data = {
                    'Device': ['Windows PC', 'iPhone 13', 'MacBook Pro', 'iPad'],
                    'Usage': [45, 30, 15, 10]
                }
                df_devices = pd.DataFrame(device_data)
                st.bar_chart(df_devices.set_index('Device'))

                # Location History
                st.subheader("📍 Location History")
                locations = [
                    {"name": "Home Office", "frequency": "65%", "risk": "Low"},
                    {"name": "Downtown Branch", "frequency": "25%", "risk": "Low"},
                    {"name": "Coffee Shop", "frequency": "10%", "risk": "Medium"}
                ]

                for loc in locations:
                    st.markdown(
                        f"**{loc['name']}** - Frequency: {loc['frequency']} "
                        f"<span style='color: {'green' if loc['risk']=='Low' else 'orange'}'>[{loc['risk']}]</span>",
                        unsafe_allow_html=True
                    )
            else:
                # Regular user behavior monitoring with enhanced visualization
                st.info("🔍 Real-time Behavior Analysis")

                # Behavioral Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Session Duration", f"{session_status.get('duration', 0)} sec", "Active")
                    risk_level = session_status.get('risk_level', 'unknown')
                    st.metric("Risk Level", risk_level.upper(), "")

                with col2:
                    risk_score = session_status.get('risk_score', 0)
                    st.metric("Risk Score", f"{risk_score * 100:.1f}%", "")
                    anomaly_count = len(session_status.get('anomalies', []))
                    st.metric("Anomalies", str(anomaly_count), "")

                with col3:
                    activity_count = len(session_status.get('recent_activities', []))
                    st.metric("Activities", str(activity_count), "")
                    st.metric("Status", "Active", "")

                # Activity Timeline
                st.subheader("🕒 Activity Timeline")
                recent_activities = session_status.get('recent_activities', [])
                if recent_activities:
                    for activity in recent_activities:
                        st.markdown(f"✓ {activity}")

                # Activity Score Timeline
                if 'activity_timeline' in session_status:
                    st.subheader("📊 Activity Score Timeline")
                    timeline = session_status['activity_timeline']

                    if 'timestamps' in timeline and 'activity_scores' in timeline:
                        df_timeline = pd.DataFrame({
                            'Time': timeline['timestamps'],
                            'Score': timeline['activity_scores']
                        })
                        st.line_chart(df_timeline.set_index('Time'))
                        st.caption("Activity Score Trend (Higher is Better)")

                # Risk Components
                st.subheader("⚠️ Risk Components")
                risk_factors = session_status.get('risk_factors', {})
                if risk_factors:
                    df_risks = pd.DataFrame({
                        'Factor': list(risk_factors.keys()),
                        'Score': list(risk_factors.values())
                    })
                    st.bar_chart(df_risks.set_index('Factor'))
                    st.caption("Risk Factor Analysis (Lower is Better)")

                # Anomaly Detection
                st.subheader("🔍 Anomaly Detection")
                anomalies = session_status.get('anomalies', [])
                if anomalies:
                    for anomaly in anomalies:
                        st.warning(f"⚠️ {anomaly}")
                else:
                    st.success("✅ No anomalies detected")

                # Location Information
                location_info = get_client_location()
                if location_info:
                    st.subheader("📍 Location Information")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                            - 🌆 **City:** {location_info['city']}
                            - 🗺️ **Region:** {location_info['region']}
                            - 🌍 **Country:** {location_info['country']}
                        """)

                    with col2:
                        st.markdown(f"""
                            - 🔒 **Location Risk:** {risk_factors.get('Location', 0) * 100:.1f}%
                            - 🏠 **Status:** {'Verified' if risk_factors.get('Location', 0) < 0.3 else 'Under Review'}
                        """)
        else:
            st.error("Failed to fetch session status")

def show_fraud_detection():
    """Display the fraud detection page"""
    st.markdown("## 🛡️ Fraud Detection")

    # Get fraud check results
    session_id = st.session_state.get('session_id')
    user_id = st.session_state.get('user_id')

    # Demo user data
    demo_user = user_id == "demo"

    if session_id:
        # Get location info for context
        location_info = get_client_location()

        fraud_check = make_api_request(
            endpoint=f"/api/fraud/check?session_id={session_id}",
            method="POST",
            data={'location_info': location_info} if location_info else None
        )

        if fraud_check:
            if demo_user:
                st.info("🔒 Real-time Fraud Detection Analysis")

                # Overall Security Score
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Security Score", "92/100", "↑ 3")
                    st.metric("Risk Level", "Low", "")

                with col2:
                    st.metric("Threat Detection", "No Threats", "✓")
                    st.metric("Authentication Strength", "Strong", "↑")

                with col3:
                    st.metric("Session Trust", "95%", "↑ 2%")
                    st.metric("Location Trust", "Verified", "")

                # Security Timeline
                st.subheader("🕒 Security Events Timeline")
                security_events = {
                    'time': pd.date_range(end=datetime.now(), periods=8, freq='1h'),
                    'event': [
                        "Two-factor authentication successful",
                        "Location verified - Home Office",
                        "Device fingerprint matched",
                        "Normal transaction pattern detected",
                        "Security settings reviewed",
                        "Password strength - Strong",
                        "Biometric authentication used",
                        "Session activity normal"
                    ],
                    'status': [
                        "secure", "secure", "secure", "secure",
                        "secure", "secure", "secure", "secure"
                    ]
                }
                df_events = pd.DataFrame(security_events)

                for idx, row in df_events.iterrows():
                    color = "green" if row['status'] == "secure" else "red"
                    st.markdown(
                        f"**{row['time'].strftime('%H:%M')}** - {row['event']} "
                        f"<span style='color: {color}'>[{row['status'].upper()}]</span>",
                        unsafe_allow_html=True
                    )

                # Threat Analysis
                st.subheader("🎯 Threat Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Security Metrics
                    security_data = {
                        'metric': ['Authentication', 'Device Trust', 'Network', 'Location', 'Behavior'],
                        'score': [95, 92, 88, 90, 94]
                    }
                    df_security = pd.DataFrame(security_data)
                    st.bar_chart(df_security.set_index('metric'))
                    st.caption("Security Component Analysis")

                with col2:
                    # Risk Trends
                    risk_data = {
                        'time': range(10),
                        'risk_score': [8, 7, 9, 8, 6, 7, 8, 7, 6, 5]
                    }
                    df_risk = pd.DataFrame(risk_data)
                    st.line_chart(df_risk.set_index('time'))
                    st.caption("Risk Score Trend (Lower is Better)")

                # Location Security
                st.subheader("📍 Location Security")
                st.markdown("""
                    - 🏠 **Current Location**: Home Office (Trusted)
                    - 🌍 **Location History**: 98% from trusted locations
                    - 🔄 **Travel Pattern**: Consistent with history
                    - ⚠️ **Unusual Activity**: None detected
                """)

                # Device Security
                st.subheader("💻 Device Security")
                device_security = {
                    'Device': ['Windows PC', 'iPhone 13', 'MacBook Pro', 'iPad'],
                    'Trust Score': [95, 92, 90, 88]
                }
                df_devices = pd.DataFrame(device_security)
                st.bar_chart(df_devices.set_index('Device'))

                # Recent Security Alerts
                st.subheader("⚠️ Recent Security Alerts")
                alerts = [
                    {"time": "2 hours ago", "message": "Login from new IP address - Verified", "level": "Info"},
                    {"time": "Yesterday", "message": "Password change successful", "level": "Info"},
                    {"time": "3 days ago", "message": "Security settings updated", "level": "Info"}
                ]

                for alert in alerts:
                    color = "green" if alert['level'] == "Info" else "red"
                    st.markdown(
                        f"**{alert['time']}** - {alert['message']} "
                        f"<span style='color: {color}'>[{alert['level']}]</span>",
                        unsafe_allow_html=True
                    )

                # Security Recommendations
                st.subheader("💡 Security Recommendations")
                st.markdown("""
                    1. ✅ Two-factor authentication is enabled
                    2. ✅ Using strong password
                    3. ✅ Regular security audits performed
                    4. ✅ Device verification active
                    5. ✅ Location monitoring enabled
                """)

            else:
                # Regular user fraud detection with enhanced visualization
                st.info("🔒 Real-time Security Analysis")

                # Security Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    security_score = 1 - fraud_check.get('risk_score', 0)
                    st.metric("Security Score", f"{security_score * 100:.1f}/100", "")
                    st.metric("Risk Level", fraud_check.get('risk_level', 'unknown').upper(), "")

                with col2:
                    anomaly_score = fraud_check.get('anomaly_score', 0)
                    st.metric("Anomaly Score", f"{anomaly_score * 100:.1f}%", "")
                    st.metric("Threats", str(len(fraud_check.get('active_threats', []))), "")

                with col3:
                    device_verified = fraud_check.get('device_verified', False)
                    st.metric("Device Status", "Verified" if device_verified else "New Device", "")
                    location_verified = fraud_check.get('location_verified', False)
                    st.metric("Location", "Verified" if location_verified else "Unverified", "")

                # Activity Timeline
                if 'activity_timeline' in fraud_check:
                    st.subheader("📊 Activity Score Timeline")
                    timeline = fraud_check['activity_timeline']

                    if 'timestamps' in timeline and 'activity_scores' in timeline:
                        df_timeline = pd.DataFrame({
                            'Time': timeline['timestamps'],
                            'Score': timeline['activity_scores']
                        })
                        st.line_chart(df_timeline.set_index('Time'))
                        st.caption("Security Score Trend (Higher is Better)")

                # Risk Components
                st.subheader("⚠️ Risk Analysis")
                risk_factors = fraud_check.get('risk_factors', {})
                if risk_factors:
                    df_risks = pd.DataFrame({
                        'Factor': list(risk_factors.keys()),
                        'Score': list(risk_factors.values())
                    })
                    st.bar_chart(df_risks.set_index('Factor'))
                    st.caption("Risk Factor Analysis (Lower is Better)")

                # Security Events
                st.subheader("🔍 Security Events")
                events = fraud_check.get('security_events', [])
                if events:
                    for event in events:
                        st.markdown(f"✓ {event}")

                # Active Threats
                threats = fraud_check.get('active_threats', [])
                if threats:
                    st.subheader("🚨 Active Threats")
                    for threat in threats:
                        st.error(f"⚠️ {threat}")
                else:
                    st.success("✅ No active threats detected")

                # Location Security
                if location_info:
                    st.subheader("📍 Location Security")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                            - 🌆 **City:** {location_info['city']}
                            - 🗺️ **Region:** {location_info['region']}
                            - 🌍 **Country:** {location_info['country']}
                            - 🔒 **Status:** {'✅ Verified' if location_verified else '⚠️ Unverified'}
                        """)

                    with col2:
                        st.markdown(f"""
                            - 🔒 **Location Risk:** {risk_factors.get('Location', 0) * 100:.1f}%
                            - 🏠 **Trust Level:** {'High' if risk_factors.get('Location', 0) < 0.3 else 'Medium'}
                            - 🔄 **Last Verified:** Just now
                            - ⚠️ **Alerts:** {'None' if not threats else len(threats)}
                        """)

                # Security Recommendations
                st.subheader("💡 Security Recommendations")
                recommendations = [
                    "✅ Keep monitoring your account activity",
                    "✅ Report any suspicious transactions immediately",
                    "✅ Enable two-factor authentication if not already enabled",
                    "✅ Review your security settings regularly"
                ]
                for rec in recommendations:
                    st.info(rec)
        else:
            st.error("Failed to fetch fraud detection results")

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Sidebar navigation
    st.sidebar.title("Navigation")

    if not st.session_state.authenticated:
        show_login_page()
    else:
        page = st.sidebar.radio(
            "Select Page",
            ["Account Overview", "Behavior Monitor", "Fraud Detection"]
        )

        if page == "Account Overview":
            show_account_overview()
        elif page == "Behavior Monitor":
            show_behavior_monitor()
        elif page == "Fraud Detection":
            show_fraud_detection()

if __name__ == "__main__":
    main()
