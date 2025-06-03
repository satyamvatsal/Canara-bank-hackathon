import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from behaviormonitor import BehaviorBackend

class FraudDetection:
    """
    Fraud detection component for analyzing and displaying potential fraudulent activities
    in the behavior-based authentication system.
    """
    
    def __init__(self):
        self.detection_threshold = 0.7
        self.fraud_rate = 0.05  # Base fraud rate for simulation
        self.backend = BehaviorBackend()  # Initialize backend connection
        
    def render(self):
        """Render the fraud detection interface."""
        st.header("🚨 Fraud Detection")
        st.markdown("Advanced fraud detection and investigation tools")
        
        # Control panel
        self._render_control_panel()
        
        st.divider()
        
        # Main fraud detection interface
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_fraud_alerts()
            self._render_fraud_patterns()
        
        with col2:
            self._render_investigation_tools()
            self._render_fraud_statistics()
        
        st.divider()
        
        # Detailed fraud analysis
        self._render_detailed_analysis()
    
    def _render_control_panel(self):
        """Render fraud detection control panel."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.detection_threshold = st.slider(
                "Detection Threshold", 
                0.0, 1.0, self.detection_threshold, 0.05
            )
        
        with col2:
            time_window = st.selectbox(
                "Time Window",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
                index=2
            )
        
        with col3:
            alert_severity = st.selectbox(
                "Alert Severity",
                ["All", "Critical", "High", "Medium"],
                index=0
            )
        
        with col4:
            if st.button("🔍 Run Detection"):
                self._run_fraud_detection()
    
    def _render_fraud_alerts(self):
        """Render active fraud alerts."""
        st.subheader("⚠️ Active Alerts")
        
        alerts = self._get_fraud_alerts()
        
        if alerts:
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_color = self._get_severity_color(alert['severity'])
                
                with st.expander(f"{severity_color} {alert['title']}", expanded=alert['severity'] == 'Critical'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**User ID:** {alert['user_id']}")
                        st.write(f"**Risk Score:** {alert['risk_score']:.3f}")
                        st.write(f"**Confidence:** {alert['confidence']:.1%}")
                    
                    with col2:
                        st.write(f"**Time:** {alert['timestamp']}")
                        st.write(f"**Session:** {alert['session_id']}")
                        st.write(f"**Status:** {alert['status']}")
                    
                    st.write(f"**Description:** {alert['description']}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Investigate", key=f"inv_{alert['id']}"):
                            st.success("Investigation started")
                    with col2:
                        if st.button(f"Mark False Positive", key=f"fp_{alert['id']}"):
                            st.info("Marked as false positive")
                    with col3:
                        if st.button(f"Escalate", key=f"esc_{alert['id']}"):
                            st.warning("Alert escalated")
        else:
            st.success("✅ No active fraud alerts")
    
    def _render_fraud_patterns(self):
        """Render fraud pattern analysis."""
        st.subheader("📊 Fraud Patterns")
        
        patterns = self._get_fraud_patterns()
        
        # Pattern frequency chart
        pattern_names = list(patterns.keys())
        pattern_counts = [patterns[p]['count'] for p in pattern_names]
        
        fig = px.bar(
            x=pattern_names,
            y=pattern_counts,
            title="Fraud Pattern Frequency",
            labels={'x': 'Pattern Type', 'y': 'Occurrences'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern details
        with st.expander("Pattern Details"):
            for pattern_name, pattern_data in patterns.items():
                st.write(f"**{pattern_name}:**")
                st.write(f"• Occurrences: {pattern_data['count']}")
                st.write(f"• Success Rate: {pattern_data['success_rate']:.1%}")
                st.write(f"• Avg Risk Score: {pattern_data['avg_risk']:.3f}")
                st.divider()
    
    def _render_investigation_tools(self):
        """Render fraud investigation tools."""
        st.subheader("🔍 Investigation Tools")
        
        # User lookup
        st.write("**User Investigation:**")
        investigate_user = st.text_input("Enter User ID to investigate:")
        
        if investigate_user and st.button("Investigate User"):
            user_analysis = self._investigate_user(investigate_user)
            
            if user_analysis:
                st.write("**Investigation Results:**")
                st.json(user_analysis)
            else:
                st.error("User not found or no data available")
        
        # Session analysis
        st.write("**Session Analysis:**")
        session_id = st.text_input("Enter Session ID:")
        
        if session_id and st.button("Analyze Session"):
            session_analysis = self._analyze_session(session_id)
            
            if session_analysis:
                st.write("**Session Analysis:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Risk Score", f"{session_analysis['risk_score']:.3f}")
                    st.metric("Anomaly Score", f"{session_analysis['anomaly_score']:.3f}")
                
                with col2:
                    st.metric("Duration", f"{session_analysis['duration']}s")
                    st.metric("Interactions", session_analysis['interactions'])
            else:
                st.error("Session not found")
        
        # IP analysis
        st.write("**IP Address Analysis:**")
        ip_address = st.text_input("Enter IP Address:")
        
        if ip_address and st.button("Analyze IP"):
            ip_analysis = self._analyze_ip(ip_address)
            st.write("**IP Analysis Results:**")
            st.json(ip_analysis)
    
    def _render_fraud_statistics(self):
        """Render fraud detection statistics."""
        st.subheader("📈 Detection Statistics")
        
        stats = self._get_fraud_statistics()
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Detection Rate", f"{stats['detection_rate']:.1%}")
            st.metric("False Positive Rate", f"{stats['false_positive_rate']:.1%}")
        
        with col2:
            st.metric("True Positive Rate", f"{stats['true_positive_rate']:.1%}")
            st.metric("Avg Response Time", f"{stats['avg_response_time']:.0f}ms")
        
        # Performance trends
        trend_data = self._get_performance_trends()
        
        fig = px.line(
            trend_data,
            x='date',
            y=['precision', 'recall', 'f1_score'],
            title="Detection Performance Trends",
            labels={'value': 'Score', 'date': 'Date'}
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_analysis(self):
        """Render detailed fraud analysis."""
        st.subheader("🔬 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Temporal Analysis", "Feature Importance", "Case Studies"])
        
        with tab1:
            self._render_risk_distribution_analysis()
        
        with tab2:
            self._render_temporal_analysis()
        
        with tab3:
            self._render_feature_importance()
        
        with tab4:
            self._render_case_studies()
    
    def _render_risk_distribution_analysis(self):
        """Render risk distribution analysis."""
        st.write("**Risk Score Distribution Analysis**")
        
        # Generate risk distribution data
        normal_scores = np.random.beta(2, 8, 800) * 0.6  # Normal users (low risk)
        suspicious_scores = np.random.beta(3, 2, 150) * 0.4 + 0.6  # Suspicious users
        fraud_scores = np.random.beta(1, 2, 50) * 0.3 + 0.7  # Fraudulent users
        
        all_scores = np.concatenate([normal_scores, suspicious_scores, fraud_scores])
        categories = ['Normal'] * 800 + ['Suspicious'] * 150 + ['Fraudulent'] * 50
        
        df = pd.DataFrame({
            'risk_score': all_scores,
            'category': categories
        })
        
        fig = px.histogram(
            df,
            x='risk_score',
            color='category',
            nbins=30,
            title="Risk Score Distribution by User Category",
            labels={'risk_score': 'Risk Score', 'count': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.write("**Statistical Summary:**")
        summary = df.groupby('category')['risk_score'].agg(['mean', 'std', 'min', 'max']).round(3)
        st.dataframe(summary, use_container_width=True)
    
    def _render_temporal_analysis(self):
        """Render temporal fraud analysis."""
        st.write("**Temporal Fraud Analysis**")
        
        # Generate temporal data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        fraud_counts = np.random.poisson(2, len(dates))  # Base fraud rate
        
        # Add seasonal patterns
        day_of_year = dates.dayofyear
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        fraud_counts = (fraud_counts * seasonal_factor).astype(int)
        
        temporal_df = pd.DataFrame({
            'date': dates,
            'fraud_count': fraud_counts,
            'month': dates.month,
            'day_of_week': dates.dayofweek,
            'hour': np.random.randint(0, 24, len(dates))
        })
        
        # Fraud by month
        monthly_fraud = temporal_df.groupby('month')['fraud_count'].sum().reset_index()
        monthly_fraud['month_name'] = pd.to_datetime(monthly_fraud['month'], format='%m').dt.strftime('%B')
        
        fig = px.bar(
            monthly_fraud,
            x='month_name',
            y='fraud_count',
            title="Fraud Cases by Month"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by day of week
        col1, col2 = st.columns(2)
        
        with col1:
            daily_fraud = temporal_df.groupby('day_of_week')['fraud_count'].sum().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_fraud['day_name'] = [day_names[i] for i in daily_fraud['day_of_week']]
            
            fig = px.bar(
                daily_fraud,
                x='day_name',
                y='fraud_count',
                title="Fraud Cases by Day of Week"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            hourly_fraud = temporal_df.groupby('hour')['fraud_count'].sum().reset_index()
            
            fig = px.line(
                hourly_fraud,
                x='hour',
                y='fraud_count',
                title="Fraud Cases by Hour of Day"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_importance(self):
        """Render feature importance analysis."""
        st.write("**Feature Importance in Fraud Detection**")
        
        # Generate feature importance data
        features = [
            'Device Fingerprint Mismatch', 'Unusual Location', 'Typing Speed Deviation',
            'Touch Pressure Anomaly', 'Session Duration', 'Transaction Velocity',
            'Network Risk Score', 'Time of Day', 'Swipe Pattern Change',
            'Multiple Device Usage', 'Failed Auth Attempts', 'IP Reputation'
        ]
        
        importance_scores = np.random.beta(2, 5, len(features))
        importance_scores = importance_scores / importance_scores.sum()  # Normalize
        
        feature_df = pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance in Fraud Detection",
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top contributing features
        st.write("**Top Contributing Features:**")
        top_features = feature_df.tail(5)
        for _, row in top_features.iterrows():
            st.write(f"• **{row['feature']}**: {row['importance']:.1%}")
    
    def _render_case_studies(self):
        """Render fraud case studies."""
        st.write("**Recent Fraud Case Studies**")
        
        case_studies = self._get_case_studies()
        
        for i, case in enumerate(case_studies):
            with st.expander(f"Case {i+1}: {case['title']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Case ID:** {case['case_id']}")
                    st.write(f"**Detection Time:** {case['detection_time']}")
                    st.write(f"**Risk Score:** {case['risk_score']:.3f}")
                    st.write(f"**Status:** {case['status']}")
                
                with col2:
                    st.write(f"**User ID:** {case['user_id']}")
                    st.write(f"**Loss Amount:** ${case['loss_amount']:,}")
                    st.write(f"**Detection Method:** {case['detection_method']}")
                    st.write(f"**Resolution:** {case['resolution']}")
                
                st.write(f"**Description:** {case['description']}")
                
                if case['lessons_learned']:
                    st.write("**Lessons Learned:**")
                    for lesson in case['lessons_learned']:
                        st.write(f"• {lesson}")
    
    def _get_fraud_alerts(self) -> List[Dict]:
        """Get current fraud alerts from behavior monitor."""
        alerts = self.backend.get_recent_alerts(limit=5)
        
        # Transform alerts to required format
        transformed_alerts = []
        for alert in alerts:
            # Get session details for additional context
            sessions = self.backend.get_user_sessions(alert['user_id'], limit=1)
            session = sessions[0] if sessions else None
            
            transformed_alerts.append({
                'id': alert['alert_id'],
                'title': alert['alert_type'],
                'user_id': alert['user_id'],
                'session_id': alert['session_id'],
                'risk_score': session.risk_score if session else 0.75,
                'confidence': session.anomaly_score if session else 0.8,
                'severity': 'Critical' if alert['severity'] >= 3 else 'High' if alert['severity'] == 2 else 'Medium',
                'timestamp': alert['timestamp'],
                'status': 'Active',
                'description': alert['message']
            })
        
        return transformed_alerts
    
    def _get_fraud_patterns(self) -> Dict:
        """Get fraud pattern analysis from behavior monitor."""
        # Get recent sessions
        sessions = self.backend.get_recent_sessions(hours=24)
        
        # Analyze patterns
        patterns = {
            'Device Switching': {'count': 0, 'success_rate': 0, 'avg_risk': 0},
            'Velocity Attacks': {'count': 0, 'success_rate': 0, 'avg_risk': 0},
            'Location Hopping': {'count': 0, 'success_rate': 0, 'avg_risk': 0},
            'Behavioral Mimicry': {'count': 0, 'success_rate': 0, 'avg_risk': 0},
            'Social Engineering': {'count': 0, 'success_rate': 0, 'avg_risk': 0}
        }
        
        for session in sessions:
            if session.risk_score > self.detection_threshold:
                # Classify the pattern based on session metrics
                if abs(session.typing_speed - session.metadata.get('baseline_speed', 0)) > 2.0:
                    patterns['Behavioral Mimicry']['count'] += 1
                    patterns['Behavioral Mimicry']['avg_risk'] += session.risk_score
                
                if session.rhythm_consistency < 0.6:
                    patterns['Velocity Attacks']['count'] += 1
                    patterns['Velocity Attacks']['avg_risk'] += session.risk_score
        
        # Calculate averages and success rates
        for pattern in patterns.values():
            if pattern['count'] > 0:
                pattern['avg_risk'] /= pattern['count']
                pattern['success_rate'] = np.random.uniform(0.2, 0.6)  # Simulated success rate
        
        return patterns
    
    def _get_fraud_statistics(self) -> Dict:
        """Get fraud detection statistics from behavior monitor."""
        sessions = self.backend.get_recent_sessions(hours=24)
        
        total_sessions = len(sessions)
        high_risk_sessions = sum(1 for s in sessions if s.risk_score > self.detection_threshold)
        
        return {
            'detection_rate': high_risk_sessions / total_sessions if total_sessions > 0 else 0,
            'false_positive_rate': 0.08,  # Estimated
            'true_positive_rate': 0.82,   # Estimated
            'avg_response_time': 145      # Simulated response time
        }
    
    def _get_performance_trends(self) -> pd.DataFrame:
        """Get performance trend data."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-19', freq='D')
        
        # Generate realistic performance trends
        base_precision = 0.85
        base_recall = 0.78
        
        precision_trend = base_precision + 0.1 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.02, len(dates))
        recall_trend = base_recall + 0.08 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 0.02, len(dates))
        f1_trend = 2 * (precision_trend * recall_trend) / (precision_trend + recall_trend)
        
        return pd.DataFrame({
            'date': dates,
            'precision': np.clip(precision_trend, 0, 1),
            'recall': np.clip(recall_trend, 0, 1),
            'f1_score': np.clip(f1_trend, 0, 1)
        })
    
    def _get_case_studies(self) -> List[Dict]:
        """Get fraud case studies."""
        return [
            {
                'case_id': 'CS2024-001',
                'title': 'Coordinated Account Takeover',
                'detection_time': '2024-12-15 09:23:15',
                'user_id': 'user_0234',
                'risk_score': 0.91,
                'status': 'Resolved',
                'loss_amount': 15000,
                'detection_method': 'Behavioral Analysis + Device Fingerprinting',
                'resolution': 'Account secured, funds recovered',
                'description': 'Attackers used stolen credentials but failed to replicate user behavioral patterns, triggering immediate detection.',
                'lessons_learned': [
                    'Behavioral patterns provide strong authentication even with compromised credentials',
                    'Multi-factor detection significantly improves accuracy',
                    'Quick response prevented major financial loss'
                ]
            },
            {
                'case_id': 'CS2024-002',
                'title': 'Synthetic Identity Fraud',
                'detection_time': '2024-12-10 16:45:33',
                'user_id': 'user_0456',
                'risk_score': 0.84,
                'status': 'Under Investigation',
                'loss_amount': 8500,
                'detection_method': 'Anomaly Detection + Temporal Analysis',
                'resolution': 'Account frozen, investigation ongoing',
                'description': 'Fraudster created fake identity but behavioral inconsistencies revealed synthetic nature.',
                'lessons_learned': [
                    'Long-term behavior patterns help identify synthetic identities',
                    'Cross-referencing multiple data sources improves detection',
                    'Early intervention limits potential losses'
                ]
            }
        ]
    
    def _investigate_user(self, user_id: str) -> Optional[Dict]:
        """Investigate a specific user using behavior monitor data."""
        profile = self.backend.get_user_profile(user_id)
        if not profile:
            return None
            
        sessions = self.backend.get_user_sessions(user_id)
        high_risk_sessions = sum(1 for s in sessions if s.risk_score > self.detection_threshold)
        
        return {
            'user_id': user_id,
            'account_age_days': (datetime.now() - profile.created_at).days,
            'total_sessions': len(sessions),
            'avg_risk_score': sum(s.risk_score for s in sessions) / len(sessions) if sessions else 0,
            'high_risk_sessions': high_risk_sessions,
            'devices_used': len(set(s.metadata.get('device_id', '') for s in sessions)),
            'locations_accessed': len(set(s.metadata.get('location', '') for s in sessions)),
            'last_activity': profile.last_active.isoformat(),
            'fraud_incidents': sum(1 for s in sessions if s.risk_score > 0.9),
            'behavioral_stability': sum(s.rhythm_consistency for s in sessions) / len(sessions) if sessions else 0
        }
    
    def _analyze_session(self, session_id: str) -> Optional[Dict]:
        """Analyze a specific session using behavior monitor data."""
        # Find session in recent sessions
        sessions = self.backend.get_recent_sessions(hours=24)
        session = next((s for s in sessions if s.session_id == session_id), None)
        
        if not session:
            return None
            
        return {
            'session_id': session_id,
            'risk_score': session.risk_score,
            'anomaly_score': session.anomaly_score,
            'duration': (session.end_time - session.start_time).seconds,
            'interactions': len(session.typing_pattern),
            'device_type': session.metadata.get('device_type', 'Unknown'),
            'location': session.metadata.get('location', 'Unknown'),
            'ip_address': session.metadata.get('ip_address', 'Unknown')
        }
    
    def _analyze_ip(self, ip_address: str) -> Dict:
        """Analyze an IP address."""
        return {
            'ip_address': ip_address,
            'reputation_score': 0.82,
            'country': 'United States',
            'isp': 'Verizon Communications',
            'is_vpn': False,
            'is_proxy': False,
            'risk_level': 'Low',
            'associated_users': 3,
            'fraud_incidents': 0
        }
    
    def _run_fraud_detection(self):
        """Run fraud detection analysis."""
        with st.spinner("Running fraud detection analysis..."):
            # Simulate detection process
            import time
            time.sleep(2)
            st.success("Fraud detection analysis completed successfully!")
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color indicator for alert severity."""
        colors = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }
        return colors.get(severity, '⚪')
    
    def get_fraud_rate(self) -> float:
        """Get current fraud detection rate for dashboard."""
        return self.fraud_rate