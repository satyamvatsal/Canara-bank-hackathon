import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from behaviormonitor import BehaviorBackend
import logging
from pydantic import BaseModel, validator
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Input validation models
class FraudMetrics(BaseModel):
    user_id: str
    session_id: str
    timestamp: datetime
    typing_speed: float
    key_intervals: List[float]
    backspace_frequency: float
    error_rate: float
    device_id: str
    location: Dict[str, str]
    
    @validator('typing_speed')
    def validate_typing_speed(cls, v):
        if v < 0 or v > 1000:  # Unrealistic typing speeds
            raise ValueError("Invalid typing speed")
        return v
    
    @validator('key_intervals')
    def validate_intervals(cls, v):
        if not v or any(i < 0 or i > 5000 for i in v):  # Unrealistic intervals
            raise ValueError("Invalid key intervals")
        return v
    
    @validator('backspace_frequency')
    def validate_backspace(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Backspace frequency must be between 0 and 1")
        return v
    
    @validator('error_rate')
    def validate_error_rate(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Error rate must be between 0 and 1")
        return v

class FraudDetection:
    """
    Fraud detection component for analyzing and displaying potential fraudulent activities
    in the behavior-based authentication system.
    """
    
    def __init__(self):
        self.detection_threshold = 0.7
        self.fraud_rate = 0.05  # Base fraud rate for simulation
        self.backend = BehaviorBackend()  # Initialize backend connection
        self.cache_timeout = 300  # 5 minutes cache timeout
        
    def analyze_typing_pattern(self, metrics: FraudMetrics) -> float:
        """
        Analyze typing pattern for potential fraud.
        
        Args:
            metrics: Validated typing metrics
            
        Returns:
            Risk score between 0 and 1
            
        Raises:
            ValueError: If metrics are invalid
        """
        try:
            # Get user's baseline profile
            user_profile = self.backend.get_user_profile(metrics.user_id)
            if not user_profile:
                logger.warning(f"No baseline profile found for user {metrics.user_id}")
                return 0.8  # High risk for unknown users
            
            # Calculate typing pattern deviation
            speed_deviation = abs(metrics.typing_speed - user_profile.baseline_typing_speed)
            speed_risk = min(speed_deviation / user_profile.baseline_typing_speed, 1.0)
            
            # Calculate rhythm deviation
            rhythm_deviation = self._calculate_rhythm_deviation(
                metrics.key_intervals,
                user_profile.baseline_rhythm
            )
            
            # Calculate error pattern risk
            error_risk = self._calculate_error_risk(
                metrics.backspace_frequency,
                metrics.error_rate
            )
            
            # Weighted risk score
            risk_score = (
                0.4 * speed_risk +
                0.4 * rhythm_deviation +
                0.2 * error_risk
            )
            
            return min(risk_score, 1.0)
        except Exception as e:
            logger.error(f"Error analyzing typing pattern: {str(e)}")
            return 0.9  # High risk on error
    
    @lru_cache(maxsize=1000)
    def _calculate_rhythm_deviation(self, current: List[float], baseline: List[float]) -> float:
        """Calculate deviation from baseline rhythm pattern."""
        try:
            if not current or not baseline:
                return 1.0
            
            # Normalize patterns to same length
            length = min(len(current), len(baseline))
            current = current[:length]
            baseline = baseline[:length]
            
            # Calculate normalized deviation
            deviations = np.abs(np.array(current) - np.array(baseline))
            avg_deviation = np.mean(deviations) / np.mean(baseline)
            
            return min(avg_deviation, 1.0)
        except Exception as e:
            logger.error(f"Error calculating rhythm deviation: {str(e)}")
            return 1.0
    
    def _calculate_error_risk(self, backspace_freq: float, error_rate: float) -> float:
        """Calculate risk based on error patterns."""
        try:
            # High backspace frequency but low error rate might indicate automated correction
            if backspace_freq > 0.3 and error_rate < 0.05:
                return 0.8
            
            # Very low error rate might indicate automated input
            if error_rate < 0.01:
                return 0.7
            
            # Very high error rate might indicate unauthorized user
            if error_rate > 0.3:
                return 0.9
            
            # Normal range
            return 0.2 + (backspace_freq + error_rate) / 2
        except Exception as e:
            logger.error(f"Error calculating error risk: {str(e)}")
            return 0.8
    
    @lru_cache(maxsize=100)
    def _get_fraud_alerts(self) -> List[Dict]:
        """Get recent fraud alerts with caching."""
        try:
            alerts = []
            recent_sessions = self.backend.get_recent_sessions(hours=24)
            
            for session in recent_sessions:
                if session.risk_score > self.detection_threshold:
                    alerts.append({
                        'id': session.session_id,
                        'user_id': session.user_id,
                        'timestamp': session.end_time,
                        'risk_score': session.risk_score,
                        'title': self._get_alert_title(session),
                        'description': self._get_alert_description(session),
                        'severity': self._get_severity(session.risk_score),
                        'confidence': self._calculate_confidence(session),
                        'session_id': session.session_id,
                        'status': 'Active'
                    })
            
            return sorted(alerts, key=lambda x: x['risk_score'], reverse=True)
        except Exception as e:
            logger.error(f"Error getting fraud alerts: {str(e)}")
            return []
    
    def _get_alert_title(self, session: Any) -> str:
        """Generate alert title based on risk factors."""
        try:
            if session.risk_score > 0.9:
                return "🚨 Critical Risk Activity Detected"
            elif session.risk_score > 0.8:
                return "⚠️ High Risk Behavior Pattern"
            elif session.risk_score > 0.7:
                return "⚡ Suspicious Activity Detected"
            else:
                return "ℹ️ Unusual Behavior Pattern"
        except Exception as e:
            logger.error(f"Error generating alert title: {str(e)}")
            return "⚠️ Alert"
    
    def _get_alert_description(self, session: Any) -> str:
        """Generate detailed alert description."""
        try:
            factors = []
            
            if hasattr(session, 'typing_speed') and session.typing_speed > 0:
                if session.typing_speed > 10:
                    factors.append("Unusually fast typing speed")
                elif session.typing_speed < 0.5:
                    factors.append("Unusually slow typing speed")
            
            if hasattr(session, 'rhythm_consistency') and session.rhythm_consistency < 0.5:
                factors.append("Inconsistent typing rhythm")
            
            if hasattr(session, 'metadata'):
                metadata = session.metadata
                if metadata.get('error_rate', 0) > 0.3:
                    factors.append("High error rate")
                if metadata.get('backspace_count', 0) > 50:
                    factors.append("Excessive backspace usage")
            
            if not factors:
                factors.append("Multiple risk factors detected")
            
            return " | ".join(factors)
        except Exception as e:
            logger.error(f"Error generating alert description: {str(e)}")
            return "Multiple risk factors detected"
    
    def _get_severity(self, risk_score: float) -> str:
        """Determine alert severity based on risk score."""
        if risk_score > 0.9:
            return "Critical"
        elif risk_score > 0.8:
            return "High"
        elif risk_score > 0.7:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_confidence(self, session: Any) -> float:
        """Calculate confidence in fraud detection."""
        try:
            # Base confidence on amount of data available
            confidence = 0.5
            
            if hasattr(session, 'typing_pattern') and session.typing_pattern:
                confidence += 0.2
            
            if hasattr(session, 'metadata'):
                metadata = session.metadata
                if metadata.get('duration_minutes', 0) > 5:
                    confidence += 0.1
                if metadata.get('backspace_count', 0) > 0:
                    confidence += 0.1
            
            if hasattr(session, 'rhythm_consistency') and session.rhythm_consistency > 0:
                confidence += 0.1
            
            return min(confidence, 0.95)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
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