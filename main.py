import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class BehaviorMonitor:
    """
    Simplified behavior monitoring component focusing on typing patterns and session activity
    """

    def __init__(self):
        self.monitoring_active = False
        self.selected_user = None

    def render(self):
        """Render the behavior monitoring interface."""
        st.header("👁️ Behavior Monitoring")
        st.markdown("Track user typing patterns and session activity")

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
            if st.button("Generate Test Data"):
                self._generate_test_data()

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
        col1, col2, col3, col4 = st.columns(4)

        profile_data = self._get_user_profile_data(self.selected_user)

        with col1:
            st.metric(
                "Typing Speed",
                f"{profile_data['typing_speed']:.1f} CPS",
                delta=f"{profile_data['speed_delta']:.1f} CPS"
            )

        with col2:
            st.metric(
                "Session Count",
                profile_data['session_count'],
                delta=profile_data['session_delta']
            )

        with col3:
            st.metric(
                "Avg Risk Score",
                f"{profile_data['avg_risk']:.2f}",
                delta=f"{profile_data['risk_delta']:.2f}"
            )

        with col4:
            st.metric(
                "Last Active",
                profile_data['last_active'],
                delta="Now" if profile_data['is_active'] else "Idle"
            )

    def _render_realtime_metrics(self):
        """Render real-time behavioral metrics."""
        st.subheader("⚡ Current Session")

        # Get current session data
        current_data = self._get_current_session_data(self.selected_user)

        if current_data:
            # Display key metrics
            metrics_data = [
                {"Metric": "Typing Speed", "Value": f"{current_data['typing_speed']:.1f} CPS", "Status": current_data['speed_status']},
                {"Metric": "Typing Rhythm", "Value": f"{current_data['rhythm_consistency']:.0%}", "Status": current_data['rhythm_status']},
                {"Metric": "Session Duration", "Value": f"{current_data['duration']:.0f}s", "Status": "Normal"},
                {"Metric": "Key Press Duration", "Value": f"{current_data['key_duration']:.0f}ms", "Status": current_data['key_status']}
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
            st.info("No active session data available")

    def _render_typing_patterns(self):
        """Render typing patterns visualization."""
        st.subheader("⌨️ Typing Patterns")

        patterns_data = self._get_typing_patterns_data(self.selected_user)

        if patterns_data:
            fig = go.Figure()

            # Add baseline pattern
            fig.add_trace(go.Scatter(
                x=patterns_data['baseline']['x'],
                y=patterns_data['baseline']['y'],
                name='Baseline',
                line=dict(color='blue', width=2, dash='dot')
            ))

            # Add current pattern
            fig.add_trace(go.Scatter(
                x=patterns_data['current']['x'],
                y=patterns_data['current']['y'],
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

        anomaly_data = self._get_anomaly_data(self.selected_user)

        if anomaly_data:
            # Anomaly score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = anomaly_data['score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Anomaly Score"},
                delta = {'reference': anomaly_data['baseline']},
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
            if anomaly_data['score'] > 0.6:
                st.warning(f"⚠️ Anomaly detected: {anomaly_data['description']}")
            else:
                st.success("✅ No significant anomalies detected")
        else:
            st.info("No anomaly detection data available")

    def _render_activity_timeline(self):
        """Render activity timeline."""
        st.subheader("📅 Activity Timeline")

        timeline_data = self._get_activity_timeline_data(self.selected_user)

        if not timeline_data.empty:
            fig = px.line(
                timeline_data,
                x='time',
                y='activity',
                title="Recent Activity Pattern",
                labels={'activity': 'Activity Level', 'time': 'Time'}
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
        analysis_data = self._get_typing_analysis(self.selected_user)

        if analysis_data:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Key Statistics:**")
                st.write(f"• Average speed: {analysis_data['avg_speed']:.1f} CPS")
                st.write(f"• Speed deviation: {analysis_data['speed_dev']:.1f} CPS")
                st.write(f"• Rhythm consistency: {analysis_data['rhythm_consistency']:.0%}")

            with col2:
                st.write("**Pattern Analysis:**")
                st.write(f"• Common errors: {analysis_data['error_rate']:.1%}")
                st.write(f"• Backspace frequency: {analysis_data['backspace_freq']:.1f}/min")
                st.write(f"• Shift key usage: {analysis_data['shift_usage']:.1f}/min")

            # Typing speed distribution
            fig = px.histogram(
                x=analysis_data['speed_samples'],
                title="Typing Speed Distribution",
                labels={'x': 'Typing Speed (CPS)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No typing analysis data available")

    def _render_session_history(self):
        """Render session history analysis."""
        history_data = self._get_session_history(self.selected_user)

        if not history_data.empty:
            fig = px.bar(
                history_data,
                x='date',
                y='duration',
                color='risk_score',
                title="Recent Session History",
                labels={'duration': 'Duration (min)', 'date': 'Date'},
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No session history data available")

    def _render_overview_metrics(self):
        """Render system-wide overview metrics."""
        col1, col2, col3, col4 = st.columns(4)

        overview_data = self._get_overview_metrics()

        with col1:
            st.metric("Active Users", overview_data['active_users'])

        with col2:
            st.metric("Typing Anomalies", overview_data['typing_anomalies'])

        with col3:
            st.metric("Avg Typing Speed", f"{overview_data['avg_speed']:.1f} CPS")

        with col4:
            st.metric("Model Accuracy", f"{overview_data['model_accuracy']:.0%}")

    def _render_active_sessions(self):
        """Render active sessions overview."""
        st.subheader("🔄 Active Sessions")

        sessions_data = self._get_active_sessions_data()

        if sessions_data:
            df = pd.DataFrame(sessions_data)

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
            st.info("No active sessions")

    def _render_typing_consistency(self):
        """Render typing consistency across users."""
        st.subheader("📝 Typing Consistency")

        consistency_data = self._get_typing_consistency_data()

        if not consistency_data.empty:
            fig = px.box(
                consistency_data,
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

        alerts = self._get_recent_alerts()

        if alerts:
            for alert in alerts[-5:]:  # Show last 5 alerts
                alert_type = alert['type']
                timestamp = alert['timestamp']
                message = alert['message']

                if alert_type == "HIGH_RISK":
                    st.error(f"🔴 {timestamp}: {message}")
                elif alert_type == "TYPING_ANOMALY":
                    st.warning(f"🟡 {timestamp}: {message}")
                else:
                    st.info(f"🔵 {timestamp}: {message}")
        else:
            st.info("No recent alerts")

    def _generate_test_data(self):
        """Generate test data for demonstration."""
        st.success("✅ Generated test data for demonstration")

    # Data generation methods
    def _get_available_users(self) -> List[str]:
        """Get list of available users."""
        return [f"user_{i:03d}" for i in range(1, 11)]

    def _get_user_profile_data(self, user_id: str) -> Dict:
        """Get user profile data."""
        return {
            'typing_speed': 4.2 + np.random.uniform(-0.5, 0.5),
            'speed_delta': np.random.uniform(-0.3, 0.3),
            'session_count': np.random.randint(15, 120),
            'session_delta': np.random.randint(-3, 7),
            'avg_risk': 0.25 + np.random.uniform(-0.1, 0.2),
            'risk_delta': np.random.uniform(-0.05, 0.05),
            'last_active': "2 min ago" if np.random.rand() > 0.3 else "15 min ago",
            'is_active': np.random.rand() > 0.3
        }

    def _get_current_session_data(self, user_id: str) -> Optional[Dict]:
        """Get current session data."""
        if not self.monitoring_active:
            return None

        speed = 4.0 + np.random.uniform(-1.0, 1.0)
        speed_status = "Normal" if 3.5 <= speed <= 4.5 else "Warning" if 3.0 <= speed <= 5.0 else "Alert"

        rhythm = 0.85 + np.random.uniform(-0.15, 0.1)
        rhythm_status = "Normal" if rhythm >= 0.8 else "Warning" if rhythm >= 0.7 else "Alert"

        key_duration = 120 + np.random.uniform(-30, 50)
        key_status = "Normal" if 100 <= key_duration <= 150 else "Warning" if 80 <= key_duration <= 180 else "Alert"

        return {
            'typing_speed': speed,
            'speed_status': speed_status,
            'rhythm_consistency': rhythm,
            'rhythm_status': rhythm_status,
            'duration': 120 + np.random.uniform(-30, 60),
            'key_duration': key_duration,
            'key_status': key_status
        }

    def _get_typing_patterns_data(self, user_id: str) -> Optional[Dict]:
        """Get typing patterns data."""
        if not self.monitoring_active:
            return None

        # Generate baseline pattern (sinusoidal with some noise)
        x = np.arange(20)
        baseline_y = 100 + 20 * np.sin(x/3) + np.random.normal(0, 5, 20)

        # Generate current pattern (similar to baseline but with possible deviations)
        current_y = baseline_y + np.random.normal(0, 15, 20)

        return {
            'baseline': {'x': x, 'y': baseline_y},
            'current': {'x': x, 'y': current_y}
        }

    def _get_anomaly_data(self, user_id: str) -> Optional[Dict]:
        """Get anomaly detection data."""
        if not self.monitoring_active:
            return None

        score = np.random.uniform(0.1, 0.9)
        descriptions = [
            "Unusual typing speed variation detected",
            "Irregular typing rhythm pattern",
            "Abnormal key press duration",
            "Increased error rate detected"
        ]

        return {
            'score': score,
            'baseline': 0.25,
            'description': np.random.choice(descriptions) if score > 0.6 else "Normal patterns",
            'details': []
        }

    def _get_activity_timeline_data(self, user_id: str) -> pd.DataFrame:
        """Get activity timeline data."""
        if not self.monitoring_active:
            return pd.DataFrame()

        # Generate sample timeline data
        times = pd.date_range(
            start=datetime.now() - timedelta(hours=6),
            end=datetime.now(),
            freq='15min'
        )

        activity = np.sin(np.arange(len(times)) * 0.5 + 0.5) + np.random.normal(0, 0.1, len(times))

        return pd.DataFrame({
            'time': times,
            'activity': activity
        })

    def _get_typing_analysis(self, user_id: str) -> Optional[Dict]:
        """Get detailed typing analysis."""
        if not self.monitoring_active:
            return None

        return {
            'avg_speed': 4.2 + np.random.uniform(-0.5, 0.5),
            'speed_dev': 0.5 + np.random.uniform(0, 0.3),
            'rhythm_consistency': 0.85 + np.random.uniform(-0.1, 0.05),
            'error_rate': 0.05 + np.random.uniform(0, 0.03),
            'backspace_freq': 2.5 + np.random.uniform(0, 1.5),
            'shift_usage': 3.0 + np.random.uniform(0, 2.0),
            'speed_samples': np.random.normal(4.2, 0.5, 100)
        }

    def _get_session_history(self, user_id: str) -> pd.DataFrame:
        """Get session history data."""
        if not self.monitoring_active:
            return pd.DataFrame()

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='D'
        )

        data = []
        for date in dates:
            if np.random.rand() > 0.2:  # 80% chance of having a session each day
                data.append({
                    'date': date.date(),
                    'duration': np.random.uniform(5, 120),
                    'risk_score': np.random.uniform(0.1, 0.5)
                })

        return pd.DataFrame(data)

    def _get_overview_metrics(self) -> Dict:
        """Get system overview metrics."""
        return {
            'active_users': np.random.randint(3, 8),
            'typing_anomalies': np.random.randint(0, 5),
            'avg_speed': 4.0 + np.random.uniform(-0.3, 0.3),
            'model_accuracy': 0.85 + np.random.uniform(-0.05, 0.05)
        }

    def _get_active_sessions_data(self) -> Optional[List[Dict]]:
        """Get active sessions data."""
        sessions = []
        for i in range(np.random.randint(1, 6)):
            risk_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.25, 0.05])
            sessions.append({
                'User': f"user_{i+1:03d}",
                'Duration': f"{np.random.randint(1, 120)}m",
                'Typing Speed': f"{3.5 + np.random.uniform(0, 2):.1f} CPS",
                'Risk': risk_level,
                'Status': 'Active'
            })

        return sessions

    def _get_typing_consistency_data(self) -> pd.DataFrame:
        """Get typing consistency data across users."""
        users = self._get_available_users()
        data = []

        for user in users:
            data.append({
                'user': user,
                'consistency': 0.8 + np.random.uniform(-0.15, 0.1)
            })

        return pd.DataFrame(data)

    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts."""
        alerts = []
        for i in range(np.random.randint(0, 5)):
            alert_type = np.random.choice(['HIGH_RISK', 'TYPING_ANOMALY', 'INFO'], p=[0.1, 0.4, 0.5])
            timestamp = (datetime.now() - timedelta(minutes=np.random.randint(1, 120))).strftime('%H:%M')

            if alert_type == 'HIGH_RISK':
                message = f"High risk session for user_{np.random.randint(1, 11):03d}"
            elif alert_type == 'TYPING_ANOMALY':
                anomalies = [
                    "Unusual typing speed variation",
                    "Irregular typing rhythm",
                    "Abnormal key press duration",
                    "Increased error rate"
                ]
                message = np.random.choice(anomalies)
            else:
                message = "New session started"

            alerts.append({
                'type': alert_type,
                'timestamp': timestamp,
                'message': message
            })

        return alerts
# Initialize and run the app
if __name__ == "__main__":
    monitor = BehaviorMonitor()
    monitor.render()
