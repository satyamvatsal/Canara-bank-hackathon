import streamlit as st
from frauddetect import FraudDetection
from behaviormonitor import BehaviorMonitor
from datetime import datetime
import requests
import json

# Set page config
st.set_page_config(
    page_title="Behavior Monitoring & Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

def login(username: str, password: str):
    """Handle user login"""
    try:
        response = requests.post(
            "http://localhost:8000/api/auth/login",
            json={
                "username": username,
                "password": password,
                "email": f"{username}@example.com",  # Demo purpose
                "account_creation_date": datetime.now().isoformat(),
                "last_login": datetime.now().isoformat(),
                "account_status": "active",
                "profile_data": {}
            }
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.authenticated = True
            st.session_state.user_id = username
            st.session_state.session_id = data['session_id']
            st.session_state.access_token = data['access_token']
            return True
        return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def render_login():
    """Render login form"""
    st.markdown("## 🔒 Secure Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if login(username, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Login failed. Please try again.")

def render_dashboard():
    """Render main dashboard"""
    # Initialize components
    behavior_monitor = BehaviorMonitor()
    fraud_detector = FraudDetection()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Account Overview", "Behavior Monitor", "Fraud Detection"]
    )

    if page == "Account Overview":
        st.markdown("## 🏦 Account Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Account Details")
            st.markdown(f"**User ID:** {st.session_state.user_id}")
            st.markdown(f"**Session ID:** {st.session_state.session_id}")
            
            # Get session status
            try:
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                response = requests.get(
                    f"http://localhost:8000/api/session/status?session_id={st.session_state.session_id}",
                    headers=headers
                )
                if response.status_code == 200:
                    session_data = response.json()
                    st.markdown(f"**Session Duration:** {session_data['duration']} seconds")
            except Exception as e:
                st.error(f"Failed to fetch session status: {str(e)}")
        
        with col2:
            st.markdown("### Security Metrics")
            try:
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                response = requests.post(
                    f"http://localhost:8000/api/fraud/check?session_id={st.session_state.session_id}",
                    headers=headers
                )
                if response.status_code == 200:
                    risk_data = response.json()
                    st.metric("Risk Score", f"{risk_data['risk_score']:.2%}")
                    st.metric("Anomaly Score", f"{risk_data['anomaly_score']:.2%}")
                    
                    if risk_data['warnings']:
                        st.warning("\n".join(risk_data['warnings']))
            except Exception as e:
                st.error(f"Failed to fetch security metrics: {str(e)}")

    elif page == "Behavior Monitor":
        behavior_monitor.render()
    else:
        fraud_detector.render()

# Main app logic
def main():
    if not st.session_state.authenticated:
        render_login()
    else:
        render_dashboard()

if __name__ == "__main__":
    main() 