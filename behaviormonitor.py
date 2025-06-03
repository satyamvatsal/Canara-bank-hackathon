import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import json
from dataclasses import dataclass, asdict
import hashlib
import time

# ====================== BACKEND SYSTEM ======================

@dataclass
class UserProfile:
    """Data class for user profile information"""
    user_id: str
    name: str
    baseline_typing_speed: float  # Characters per second
    baseline_rhythm: List[float]  # List of time intervals between key presses
    baseline_key_duration: float  # Average key press duration in ms
    created_at: datetime
    last_active: datetime

@dataclass
class SessionRecord:
    """Data class for session records"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    typing_speed: float
    rhythm_consistency: float
    key_duration: float
    risk_score: float
    anomaly_score: float
    typing_pattern: List[float]
    metadata: Dict[str, Any]

class BehaviorBackend:
    """Backend system for behavior monitoring data storage and retrieval"""
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self._init_db()
        self._seed_test_data()
    
    def _init_db(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        
        # User profiles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            baseline_typing_speed REAL,
            baseline_rhythm TEXT,
            baseline_key_duration REAL,
            created_at TEXT,
            last_active TEXT
        )
        """)
        
        # Sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            start_time TEXT,
            end_time TEXT,
            typing_speed REAL,
            rhythm_consistency REAL,
            key_duration REAL,
            risk_score REAL,
            anomaly_score REAL,
            typing_pattern TEXT,
            metadata TEXT,
            FOREIGN KEY(user_id) REFERENCES user_profiles(user_id)
        )
        """)
        
        # Alerts table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            alert_type TEXT,
            message TEXT,
            timestamp TEXT,
            severity INTEGER,
            FOREIGN KEY(user_id) REFERENCES user_profiles(user_id),
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """)
        
        self.conn.commit()
    
    def _seed_test_data(self):
        """Seed the database with test data"""
        # Create test users
        for i in range(1, 11):
            user_id = f"user_{i:03d}"
            rhythm_pattern = [100 + 20 * np.sin(x/3) for x in range(20)]
            
            profile = UserProfile(
                user_id=user_id,
                name=f"User {i}",
                baseline_typing_speed=4.0 + (i % 3) * 0.5,
                baseline_rhythm=rhythm_pattern,
                baseline_key_duration=120 + (i % 5) * 10,
                created_at=datetime.now() - timedelta(days=30),
                last_active=datetime.now() - timedelta(hours=np.random.randint(0, 24))
            )
            self.save_user_profile(profile)
            
            # Create historical sessions for each user
            for day in range(30):
                if np.random.rand() > 0.3:  # 70% chance of having a session each day
                    session = self._generate_session_data(user_id, day)
                    self.save_session(session)
    
    def _generate_session_data(self, user_id: str, days_ago: int) -> SessionRecord:
        """Generate realistic session data"""
        profile = self.get_user_profile(user_id)
        
        # Base values from profile
        base_speed = profile.baseline_typing_speed
        base_rhythm = profile.baseline_rhythm
        base_key_duration = profile.baseline_key_duration
        
        # Add some variation
        speed = base_speed + np.random.uniform(-1.0, 1.0)
        rhythm_consistency = 0.8 + np.random.uniform(-0.2, 0.1)
        key_duration = base_key_duration + np.random.uniform(-30, 30)
        
        # Generate typing pattern similar to baseline but with variations
        current_pattern = [max(50, x + np.random.normal(0, 15)) for x in base_rhythm]
        
        # Calculate risk and anomaly scores
        speed_dev = abs(speed - base_speed) / base_speed
        rhythm_dev = 1 - rhythm_consistency
        key_dev = abs(key_duration - base_key_duration) / base_key_duration
        
        risk_score = min(1.0, 0.3 * speed_dev + 0.5 * rhythm_dev + 0.2 * key_dev)
        anomaly_score = min(1.0, risk_score * (1 + np.random.uniform(-0.1, 0.1)))
        
        # Session duration between 1 minute and 2 hours
        duration_minutes = np.random.uniform(1, 120)
        start_time = datetime.now() - timedelta(days=days_ago, minutes=duration_minutes)
        end_time = datetime.now() - timedelta(days=days_ago)
        
        return SessionRecord(
            session_id=f"sess_{user_id}_{days_ago}_{int(time.time())}",
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            typing_speed=speed,
            rhythm_consistency=rhythm_consistency,
            key_duration=key_duration,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            typing_pattern=current_pattern,
            metadata={
                "duration_minutes": duration_minutes,
                "error_rate": np.random.uniform(0.01, 0.1),
                "backspace_count": np.random.randint(5, 50)
            }
        )
    
    def save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO user_profiles 
        (user_id, name, baseline_typing_speed, baseline_rhythm, baseline_key_duration, created_at, last_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.user_id,
            profile.name,
            profile.baseline_typing_speed,
            json.dumps(profile.baseline_rhythm),
            profile.baseline_key_duration,
            profile.created_at.isoformat(),
            profile.last_active.isoformat()
        ))
        
        self.conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM user_profiles WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        return UserProfile(
            user_id=row[0],
            name=row[1],
            baseline_typing_speed=row[2],
            baseline_rhythm=json.loads(row[3]),
            baseline_key_duration=row[4],
            created_at=datetime.fromisoformat(row[5]),
            last_active=datetime.fromisoformat(row[6])
        )
    
    def save_session(self, session: SessionRecord):
        """Save session record to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        INSERT INTO sessions 
        (session_id, user_id, start_time, end_time, typing_speed, rhythm_consistency, key_duration, 
         risk_score, anomaly_score, typing_pattern, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.user_id,
            session.start_time.isoformat(),
            session.end_time.isoformat(),
            session.typing_speed,
            session.rhythm_consistency,
            session.key_duration,
            session.risk_score,
            session.anomaly_score,
            json.dumps(session.typing_pattern),
            json.dumps(session.metadata)
        ))
        
        # Update user's last active time
        cursor.execute("""
        UPDATE user_profiles SET last_active = ? WHERE user_id = ?
        """, (session.end_time.isoformat(), session.user_id))
        
        self.conn.commit()
        
        # Generate alerts if needed
        if session.anomaly_score > 0.7:
            self.create_alert(
                user_id=session.user_id,
                session_id=session.session_id,
                alert_type="TYPING_ANOMALY",
                message=f"High anomaly score detected ({session.anomaly_score:.2f})",
                severity=2
            )
    
    def get_user_sessions(self, user_id: str, limit: int = 100) -> List[SessionRecord]:
        """Get sessions for a specific user"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM sessions 
        WHERE user_id = ?
        ORDER BY end_time DESC
        LIMIT ?
        """, (user_id, limit))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(SessionRecord(
                session_id=row[0],
                user_id=row[1],
                start_time=datetime.fromisoformat(row[2]),
                end_time=datetime.fromisoformat(row[3]),
                typing_speed=row[4],
                rhythm_consistency=row[5],
                key_duration=row[6],
                risk_score=row[7],
                anomaly_score=row[8],
                typing_pattern=json.loads(row[9]),
                metadata=json.loads(row[10])
            ))
        
        return sessions
    
    def get_recent_sessions(self, hours: int = 24) -> List[SessionRecord]:
        """Get recent sessions across all users"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
        SELECT * FROM sessions 
        WHERE end_time > ?
        ORDER BY end_time DESC
        """, (cutoff,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(SessionRecord(
                session_id=row[0],
                user_id=row[1],
                start_time=datetime.fromisoformat(row[2]),
                end_time=datetime.fromisoformat(row[3]),
                typing_speed=row[4],
                rhythm_consistency=row[5],
                key_duration=row[6],
                risk_score=row[7],
                anomaly_score=row[8],
                typing_pattern=json.loads(row[9]),
                metadata=json.loads(row[10])
            ))
        
        return sessions
    
    def create_alert(self, user_id: str, session_id: str, alert_type: str, 
                    message: str, severity: int = 1):
        """Create a new alert record"""
        cursor = self.conn.cursor()
        alert_id = hashlib.md5(f"{user_id}{session_id}{time.time()}".encode()).hexdigest()
        
        cursor.execute("""
        INSERT INTO alerts 
        (alert_id, user_id, session_id, alert_type, message, timestamp, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_id,
            user_id,
            session_id,
            alert_type,
            message,
            datetime.now().isoformat(),
            severity
        ))
        
        self.conn.commit()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT a.*, u.name FROM alerts a
        LEFT JOIN user_profiles u ON a.user_id = u.user_id
        ORDER BY timestamp DESC
        LIMIT ?
        """, (limit,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'alert_id': row[0],
                'user_id': row[1],
                'user_name': row[6],
                'session_id': row[2],
                'alert_type': row[3],
                'message': row[4],
                'timestamp': datetime.fromisoformat(row[5]),
                'severity': row[6]
            })
        
        return alerts
    
    def get_active_users(self) -> List[UserProfile]:
        """Get users active in the last 15 minutes"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(minutes=15)).isoformat()
        
        cursor.execute("""
        SELECT * FROM user_profiles 
        WHERE last_active > ?
        ORDER BY last_active DESC
        """, (cutoff,))
        
        users = []
        for row in cursor.fetchall():
            users.append(UserProfile(
                user_id=row[0],
                name=row[1],
                baseline_typing_speed=row[2],
                baseline_rhythm=json.loads(row[3]),
                baseline_key_duration=row[4],
                created_at=datetime.fromisoformat(row[5]),
                last_active=datetime.fromisoformat(row[6])
            ))
        
        return users
    
    def get_typing_stats(self) -> Dict[str, Any]:
        """Get aggregated typing statistics"""
        cursor = self.conn.cursor()
        
        # Get average typing speed
        cursor.execute("SELECT AVG(typing_speed) FROM sessions")
        avg_speed = cursor.fetchone()[0] or 0.0
        
        # Get count of typing anomalies
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE anomaly_score > 0.7")
        anomaly_count = cursor.fetchone()[0] or 0
        
        # Get count of active sessions in last hour
        cursor.execute("""
        SELECT COUNT(DISTINCT user_id) FROM sessions 
        WHERE end_time > ?
        """, ((datetime.now() - timedelta(hours=1)).isoformat(),))
        active_users = cursor.fetchone()[0] or 0
        
        return {
            'avg_speed': avg_speed,
            'anomaly_count': anomaly_count,
            'active_users': active_users,
            'model_accuracy': 0.85 + np.random.uniform(-0.05, 0.05)  # Simulated accuracy
        }

    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for a specific session"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
        SELECT * FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if not row:
            return {
                "duration": 0,
                "typing_speed": 0,
                "risk_score": 0,
                "anomaly_score": 0,
                "rhythm_consistency": 0
            }
            
        start_time = datetime.fromisoformat(row[2])
        end_time = datetime.fromisoformat(row[3])
        duration = (end_time - start_time).total_seconds()
        
        return {
            "duration": duration,
            "typing_speed": row[4],
            "risk_score": row[7],
            "anomaly_score": row[8],
            "rhythm_consistency": row[5]
        }

# ====================== FRONTEND COMPONENT ======================

class BehaviorMonitor:
    """
    Behavior monitoring component with backend integration
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.selected_user = None
        self.backend = BehaviorBackend()
        
    def render(self):
        """Render the behavior monitoring interface."""
        st.header("👁️ Behavior Monitoring")
        st.markdown("Track user typing patterns and session activity with persistent backend")
        
        # Control panel
        self._render_control_panel()
        
        st.divider()
        
        # Main monitoring interface
        if self.selected_user:
            self._render_user_monitoring()
        else:
            self._render_overview_monitoring()
    
    def _render_control_panel(self):
        """Render monitoring control panel."""
        col1, col2, col3 = st.columns([1,2,1])
        
        with col1:
            self.monitoring_active = st.toggle("Monitoring", value=self.monitoring_active)
        
        with col2:
            # User selection
            available_users = self._get_available_users()
            selected_user = st.selectbox(
                "Select User",
                ["All Users"] + available_users,
                index=0
            )
            self.selected_user = selected_user if selected_user != "All Users" else None
        
        with col3:
            if st.button("Simulate New Session"):
                self._simulate_new_session()
        
        # Status indicators
        col1, col2 = st.columns(2)
        
        with col1:
            status_color = "🟢" if self.monitoring_active else "🔴"
            st.write(f"**Status:** {status_color} {'Active' if self.monitoring_active else 'Inactive'}")
        
        with col2:
            if self.selected_user:
                st.write(f"**Monitoring:** {self.selected_user}")
            else:
                st.write("**Monitoring:** All Users")
    
    def _simulate_new_session(self):
        """Simulate a new user session"""
        if not self.selected_user:
            users = self._get_available_users()
            user_id = np.random.choice(users)
        else:
            user_id = self.selected_user
        
        session = self.backend._generate_session_data(user_id, 0)  # 0 days ago = today
        session.end_time = datetime.now()
        session.start_time = datetime.now() - timedelta(minutes=session.metadata['duration_minutes'])
        self.backend.save_session(session)
        
        st.success(f"✅ Simulated new session for {user_id}")
    
    def _render_user_monitoring(self):
        """Render monitoring interface for a specific user."""
        st.subheader(f"👤 User: {self.selected_user}")
        
        # User profile summary
        self._render_user_profile_summary()
        
        st.divider()
        
        # Real-time behavioral metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_realtime_metrics()
            self._render_typing_patterns()
        
        with col2:
            self._render_anomaly_detection()
            self._render_activity_timeline()
        
        st.divider()
        
        # Detailed behavioral analysis
        self._render_detailed_analysis()
    
    def _render_overview_monitoring(self):
        """Render overview monitoring for all users."""
        st.subheader("🌐 System Overview")
        
        # Overview metrics
        self._render_overview_metrics()
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_active_sessions()
            self._render_typing_consistency()
        
        with col2:
            self._render_alert_feed()
    
    def _render_user_profile_summary(self):
        """Render user behavioral profile summary."""
        profile = self.backend.get_user_profile(self.selected_user)
        if not profile:
            st.error("User profile not found")
            return
        
        # Get recent sessions for delta calculations
        sessions = self.backend.get_user_sessions(self.selected_user, limit=10)
        
        # Calculate metrics
        if len(sessions) >= 2:
            speed_delta = sessions[0].typing_speed - sessions[1].typing_speed
            risk_delta = sessions[0].risk_score - sessions[1].risk_score
        else:
            speed_delta = 0
            risk_delta = 0
        
        # Session count
        all_sessions = self.backend.get_user_sessions(self.selected_user)
        session_count = len(all_sessions)
        
        # Average risk score
        if all_sessions:
            avg_risk = sum(s.risk_score for s in all_sessions) / len(all_sessions)
        else:
            avg_risk = 0
        
        # Last active status
        last_active = profile.last_active
        is_active = (datetime.now() - last_active) < timedelta(minutes=15)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Typing Speed",
                f"{sessions[0].typing_speed:.1f} CPS" if sessions else "N/A",
                delta=f"{speed_delta:.1f} CPS" if sessions else None
            )
        
        with col2:
            st.metric(
                "Session Count",
                session_count,
                delta=f"+{len([s for s in all_sessions if (datetime.now() - s.end_time) < timedelta(days=1)])} today"
            )
        
        with col3:
            st.metric(
                "Avg Risk Score",
                f"{avg_risk:.2f}" if all_sessions else "N/A",
                delta=f"{risk_delta:.2f}" if len(sessions) >= 2 else None
            )
        
        with col4:
            delta = "Now" if is_active else "Idle"
            if not is_active:
                delta = f"{int((datetime.now() - last_active).total_seconds() / 60):.0f} min ago"
            
            st.metric(
                "Last Active",
                last_active.strftime("%Y-%m-%d %H:%M"),
                delta=delta
            )
    
    def _render_realtime_metrics(self):
        """Render real-time behavioral metrics."""
        st.subheader("⚡ Current Session")
        
        if not self.selected_user:
            st.info("Select a user to view session data")
            return
        
        sessions = self.backend.get_user_sessions(self.selected_user, limit=1)
        
        if sessions:
            current_session = sessions[0]
            
            # Determine status indicators
            profile = self.backend.get_user_profile(self.selected_user)
            
            speed_status = "Normal"
            if abs(current_session.typing_speed - profile.baseline_typing_speed) > 1.0:
                speed_status = "Warning" if abs(current_session.typing_speed - profile.baseline_typing_speed) > 1.5 else "Alert"
            
            rhythm_status = "Normal"
            if current_session.rhythm_consistency < 0.7:
                rhythm_status = "Warning" if current_session.rhythm_consistency > 0.6 else "Alert"
            
            key_status = "Normal"
            if abs(current_session.key_duration - profile.baseline_key_duration) > 30:
                key_status = "Warning" if abs(current_session.key_duration - profile.baseline_key_duration) > 50 else "Alert"
            
            # Display key metrics
            metrics_data = [
                {"Metric": "Typing Speed", "Value": f"{current_session.typing_speed:.1f} CPS", "Status": speed_status},
                {"Metric": "Typing Rhythm", "Value": f"{current_session.rhythm_consistency:.0%}", "Status": rhythm_status},
                {"Metric": "Session Duration", "Value": f"{current_session.metadata['duration_minutes']:.0f} min", "Status": "Normal"},
                {"Metric": "Key Press Duration", "Value": f"{current_session.key_duration:.0f}ms", "Status": key_status}
            ]
            
            df = pd.DataFrame(metrics_data)
            
            # Color code status
            def color_status(val):
                if val == "Normal":
                    return "background-color: #d4edda"
                elif val == "Warning":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #f8d7da"
            
            styled_df = df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No session data available for this user")
    
    def _render_typing_patterns(self):
        """Render typing patterns visualization."""
        st.subheader("⌨️ Typing Patterns")
        
        if not self.selected_user:
            st.info("Select a user to view typing patterns")
            return
        
        profile = self.backend.get_user_profile(self.selected_user)
        sessions = self.backend.get_user_sessions(self.selected_user, limit=1)
        
        if profile and sessions:
            fig = go.Figure()
            
            # Add baseline pattern
            fig.add_trace(go.Scatter(
                x=list(range(len(profile.baseline_rhythm))),
                y=profile.baseline_rhythm,
                name='Baseline',
                line=dict(color='blue', width=2, dash='dot')
            ))
            
            # Add current pattern
            fig.add_trace(go.Scatter(
                x=list(range(len(sessions[0].typing_pattern))),
                y=sessions[0].typing_pattern,
                name='Current',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Typing Rhythm Pattern",
                xaxis_title="Key Press Sequence",
                yaxis_title="Time Between Presses (ms)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No typing pattern data available")
    
    def _render_anomaly_detection(self):
        """Render real-time anomaly detection results."""
        st.subheader("🚨 Anomaly Detection")
        
        if not self.selected_user:
            st.info("Select a user to view anomaly detection")
            return
        
        sessions = self.backend.get_user_sessions(self.selected_user, limit=1)
        
        if sessions:
            current_session = sessions[0]
            
            # Anomaly score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_session.anomaly_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Anomaly Score"},
                delta = {'reference': 0.3},  # Baseline
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            if current_session.anomaly_score > 0.6:
                st.warning(f"⚠️ Anomaly detected in current session (score: {current_session.anomaly_score:.2f})")
            else:
                st.success("✅ No significant anomalies detected")
        else:
            st.info("No anomaly detection data available")
    
    def _render_activity_timeline(self):
        """Render activity timeline."""
        st.subheader("📅 Activity Timeline")
        
        if not self.selected_user:
            st.info("Select a user to view activity timeline")
            return
        
        sessions = self.backend.get_user_sessions(self.selected_user, limit=20)
        
        if sessions:
            # Prepare timeline data
            timeline_data = []
            for session in sessions:
                duration_hours = session.metadata['duration_minutes'] / 60
                timeline_data.append({
                    'time': session.end_time,
                    'activity': duration_hours * 10,  # Scale for visibility
                    'duration': duration_hours
                })
            
            df = pd.DataFrame(timeline_data)
            
            fig = px.line(
                df,
                x='time',
                y='activity',
                title="Recent Activity Pattern",
                labels={'activity': 'Activity Level', 'time': 'Time'},
                hover_data=['duration']
            )
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity timeline data available")
    
    def _render_detailed_analysis(self):
        """Render detailed behavioral analysis."""
        st.subheader("🔍 Detailed Analysis")
        
        tab1, tab2 = st.tabs(["Typing Analysis", "Session History"])
        
        with tab1:
            self._render_typing_analysis()
        
        with tab2:
            self._render_session_history()
    
    def _render_typing_analysis(self):
        """Render detailed typing analysis."""
        if not self.selected_user:
            st.info("Select a user to view typing analysis")
            return
        
        profile = self.backend.get_user_profile(self.selected_user)
        sessions = self.backend.get_user_sessions(self.selected_user, limit=50)
        
        if profile and sessions:
            # Calculate statistics
            speeds = [s.typing_speed for s in sessions]
            rhythms = [s.rhythm_consistency for s in sessions]
            key_durations = [s.key_duration for s in sessions]
            
            avg_speed = np.mean(speeds)
            speed_dev = np.std(speeds)
            avg_rhythm = np.mean(rhythms)
            
            # Error rate and other metrics from metadata
            error_rates = [s.metadata.get('error_rate', 0) for s in sessions]
            backspace_counts = [s.metadata.get('backspace_count', 0) for s in sessions]
            durations = [s.metadata.get('duration_minutes', 1) for s in sessions]
            
            avg_error_rate = np.mean(error_rates)
            backspace_freq = np.mean([b/d for b, d in zip(backspace_counts, durations)])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Statistics:**")
                st.write(f"• Average speed: {avg_speed:.1f} CPS")
                st.write(f"• Speed deviation: {speed_dev:.1f} CPS")
                st.write(f"• Rhythm consistency: {avg_rhythm:.0%}")
            
            with col2:
                st.write("**Pattern Analysis:**")
                st.write(f"• Common errors: {avg_error_rate:.1%}")
                st.write(f"• Backspace frequency: {backspace_freq:.1f}/min")
                st.write(f"• Key duration: {np.mean(key_durations):.1f}ms")
            
            # Typing speed distribution
            fig = px.histogram(
                x=speeds,
                title="Typing Speed Distribution",
                labels={'x': 'Typing Speed (CPS)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No typing analysis data available")
    
    def _render_session_history(self):
        """Render session history analysis."""
        if not self.selected_user:
            st.info("Select a user to view session history")
            return
        
        sessions = self.backend.get_user_sessions(self.selected_user, limit=30)
        
        if sessions:
            # Prepare session history data
            history_data = []
            for session in sessions:
                history_data.append({
                    'date': session.end_time.date(),
                    'duration': session.metadata['duration_minutes'],
                    'risk_score': session.risk_score,
                    'anomaly_score': session.anomaly_score
                })
            
            df = pd.DataFrame(history_data)
            
            fig = px.bar(
                df,
                x='date',
                y='duration',
                color='risk_score',
                title="Recent Session History",
                labels={'duration': 'Duration (min)', 'date': 'Date'},
                color_continuous_scale='RdYlGn_r',
                hover_data=['anomaly_score']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No session history data available")
    
    def _render_overview_metrics(self):
        """Render system-wide overview metrics."""
        stats = self.backend.get_typing_stats()
        active_users = len(self.backend.get_active_users())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Users", active_users)
        
        with col2:
            st.metric("Typing Anomalies", stats['anomaly_count'])
        
        with col3:
            st.metric("Avg Typing Speed", f"{stats['avg_speed']:.1f} CPS")
        
        with col4:
            st.metric("Model Accuracy", f"{stats['model_accuracy']:.0%}")
    
    def _render_active_sessions(self):
        """Render active sessions overview."""
        st.subheader("🔄 Active Sessions")
        
        active_users = self.backend.get_active_users()
        recent_sessions = self.backend.get_recent_sessions(hours=1)
        
        if recent_sessions:
            # Group sessions by user
            user_sessions = {}
            for session in recent_sessions:
                if session.user_id not in user_sessions:
                    user_sessions[session.user_id] = []
                user_sessions[session.user_id].append(session)
            
            # Prepare session data for display
            session_data = []
            for user_id, sessions in user_sessions.items():
                latest_session = sessions[0]
                profile = self.backend.get_user_profile(user_id)
                
                risk_level = "Low"
                if latest_session.anomaly_score > 0.7:
                    risk_level = "High"
                elif latest_session.anomaly_score > 0.5:
                    risk_level = "Medium"
                
                session_data.append({
                    'User': profile.name if profile else user_id,
                    'Duration': f"{latest_session.metadata['duration_minutes']:.0f} min",
                    'Typing Speed': f"{latest_session.typing_speed:.1f} CPS",
                    'Risk': risk_level,
                    'Status': 'Active' if (datetime.now() - latest_session.end_time) < timedelta(minutes=5) else 'Recent'
                })
            
            df = pd.DataFrame(session_data)
            
            # Color code risk levels
            def color_risk(val):
                if val == "High":
                    return "background-color: #f8d7da"
                elif val == "Medium":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #d4edda"
            
            styled_df = df.style.applymap(color_risk, subset=['Risk'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active sessions in the last hour")
    
    def _render_typing_consistency(self):
        """Render typing consistency across users."""
        st.subheader("📝 Typing Consistency")
        
        users = self._get_available_users()
        consistency_data = []
        
        for user in users:
            sessions = self.backend.get_user_sessions(user, limit=10)
            if sessions:
                avg_consistency = np.mean([s.rhythm_consistency for s in sessions])
                consistency_data.append({
                    'user': user,
                    'consistency': avg_consistency
                })
        
        if consistency_data:
            df = pd.DataFrame(consistency_data)
            
            fig = px.box(
                df,
                y='consistency',
                points="all",
                title="Typing Rhythm Consistency Across Users"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No consistency data available")
    
    def _render_alert_feed(self):
        """Render real-time alert feed."""
        st.subheader("🔔 Recent Alerts")
        
        alerts = self.backend.get_recent_alerts(limit=5)
        
        if alerts:
            for alert in alerts:
                timestamp = alert['timestamp'].strftime('%H:%M')
                message = alert['message']
                
                if alert['severity'] >= 2:
                    st.error(f"🔴 {timestamp}: {message} (User: {alert.get('user_name', alert['user_id'])})")
                elif alert['severity'] >= 1:
                    st.warning(f"🟡 {timestamp}: {message} (User: {alert.get('user_name', alert['user_id'])})")
                else:
                    st.info(f"🔵 {timestamp}: {message} (User: {alert.get('user_name', alert['user_id'])})")
        else:
            st.info("No recent alerts")
    
    def _get_available_users(self) -> List[str]:
        """Get list of available users from backend."""
        return [user.user_id for user in self.backend.get_active_users()] or [f"user_{i:03d}" for i in range(1, 6)]

# Initialize and run the app
if __name__ == "__main__":
    monitor = BehaviorMonitor()
    monitor.render()