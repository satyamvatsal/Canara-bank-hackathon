import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import json

class RiskLevel(Enum):
    """Risk level classifications for user sessions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskScorer:
    """
    Calculates comprehensive risk scores based on behavioral analysis,
    anomaly detection results, and contextual factors.
    """
    
    def __init__(self):
        # Default risk thresholds
        self.thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
        
        # Feature weights for risk calculation
        self.feature_weights = {
            'anomaly_score': 0.30,
            'behavior_deviation': 0.25,
            'temporal_factors': 0.15,
            'device_factors': 0.10,
            'contextual_factors': 0.10,
            'historical_risk': 0.10
        }
        
        # Risk multipliers for different scenarios
        self.risk_multipliers = {
            'new_device': 1.5,
            'unusual_location': 1.3,
            'unusual_time': 1.2,
            'high_value_transaction': 1.4,
            'multiple_failed_attempts': 2.0,
            'velocity_anomaly': 1.6
        }
        
        # Historical risk data
        self.user_risk_history = {}
        self.session_risk_cache = {}
        
    def calculate_risk_score(self, analysis_results: Dict, 
                           session_context: Optional[Dict] = None) -> Dict:
        """
        Calculate comprehensive risk score for a user session.
        
        Args:
            analysis_results: Results from behavior analysis and anomaly detection
            session_context: Additional contextual information about the session
            
        Returns:
            Dictionary containing detailed risk assessment
        
        Raises:
            ValueError: If analysis_results is None or empty
        """
        if not analysis_results:
            raise ValueError("Analysis results cannot be None or empty")
        
        if session_context is None:
            session_context = {}
        
        risk_components = {}
        
        # 1. Anomaly-based risk
        risk_components['anomaly_risk'] = self._calculate_anomaly_risk(analysis_results)
        
        # 2. Behavioral deviation risk
        risk_components['behavior_risk'] = self._calculate_behavior_risk(analysis_results)
        
        # 3. Temporal risk factors
        risk_components['temporal_risk'] = self._calculate_temporal_risk(session_context)
        
        # 4. Device-based risk factors
        risk_components['device_risk'] = self._calculate_device_risk(session_context)
        
        # 5. Contextual risk factors
        risk_components['contextual_risk'] = self._calculate_contextual_risk(session_context)
        
        # 6. Historical risk pattern
        risk_components['historical_risk'] = self._calculate_historical_risk(
            analysis_results.get('user_id'), session_context
        )
        
        # Validate all risk components are present
        missing_components = [
            component for component in ['anomaly_risk', 'behavior_risk', 'temporal_risk',
                                      'device_risk', 'contextual_risk', 'historical_risk']
            if component not in risk_components
        ]
        if missing_components:
            raise ValueError(f"Missing risk components: {missing_components}")
        
        # Calculate weighted risk score
        base_risk_score = sum(
            risk_components[component] * self.feature_weights[component.replace('_risk', '_factors')]
            for component in risk_components
        )
        
        # Apply risk multipliers
        multiplier = self._calculate_risk_multiplier(session_context)
        final_risk_score = min(base_risk_score * multiplier, 1.0)
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_risk_score)
        
        # Generate risk explanation
        risk_explanation = self._generate_risk_explanation(
            risk_components, multiplier, session_context
        )
        
        risk_assessment = {
            'user_id': analysis_results.get('user_id'),
            'session_id': session_context.get('session_id'),
            'timestamp': datetime.now(),
            'risk_score': final_risk_score,
            'risk_level': risk_level.value,
            'risk_components': risk_components,
            'base_score': base_risk_score,
            'risk_multiplier': multiplier,
            'risk_explanation': risk_explanation,
            'recommended_actions': self._get_recommended_actions(risk_level),
            'confidence': self._calculate_confidence(risk_components)
        }
        
        # Cache risk assessment
        if 'user_id' in analysis_results:
            self._update_risk_history(analysis_results['user_id'], risk_assessment)
        
        return risk_assessment
    
    def _calculate_anomaly_risk(self, analysis_results: Dict) -> float:
        """Calculate risk based on anomaly detection results."""
        anomaly_risk = 0.0
        
        # Check anomaly probability from detector
        if 'anomaly_probability' in analysis_results:
            anomaly_prob = analysis_results['anomaly_probability']
            if isinstance(anomaly_prob, np.ndarray):
                anomaly_risk = np.mean(anomaly_prob)
            else:
                anomaly_risk = float(anomaly_prob)
        
        # Check ensemble predictions
        elif 'ensemble_prediction' in analysis_results:
            predictions = analysis_results['ensemble_prediction']
            if isinstance(predictions, np.ndarray):
                anomaly_rate = (predictions == -1).sum() / len(predictions)
                anomaly_risk = anomaly_rate
        
        # Check individual model predictions
        elif 'model_predictions' in analysis_results:
            anomaly_rates = []
            for model, predictions in analysis_results['model_predictions'].items():
                if isinstance(predictions, np.ndarray):
                    rate = (predictions == -1).sum() / len(predictions)
                    anomaly_rates.append(rate)
            
            if anomaly_rates:
                anomaly_risk = np.mean(anomaly_rates)
        
        return min(anomaly_risk, 1.0)
    
    def _calculate_behavior_risk(self, analysis_results: Dict) -> float:
        """Calculate risk based on behavioral pattern deviations."""
        behavior_risk = 0.0
        
        # Profile match score (lower = higher risk)
        if 'profile_match_score' in analysis_results:
            profile_match = analysis_results['profile_match_score']
            behavior_risk = max(0, 1.0 - profile_match)
        
        # Feature deviations
        if 'feature_deviations' in analysis_results:
            deviations = analysis_results['feature_deviations']
            high_deviation_count = sum(
                1 for dev in deviations.values() 
                if isinstance(dev, dict) and dev.get('is_anomaly', False)
            )
            
            if deviations:
                deviation_rate = high_deviation_count / len(deviations)
                behavior_risk = max(behavior_risk, deviation_rate)
        
        # Consistency score (lower = higher risk)
        if 'consistency_score' in analysis_results:
            consistency = analysis_results['consistency_score']
            inconsistency_risk = max(0, 1.0 - consistency)
            behavior_risk = max(behavior_risk, inconsistency_risk * 0.5)
        
        # Behavioral anomalies
        if 'behavioral_anomalies' in analysis_results:
            anomaly_count = len(analysis_results['behavioral_anomalies'])
            anomaly_risk = min(anomaly_count * 0.2, 1.0)
            behavior_risk = max(behavior_risk, anomaly_risk)
        
        return min(behavior_risk, 1.0)
    
    def _calculate_temporal_risk(self, session_context: Dict) -> float:
        """Calculate risk based on temporal factors."""
        temporal_risk = 0.0
        
        current_time = datetime.now()
        
        # Unusual hour (late night/early morning transactions)
        hour = current_time.hour
        if hour < 6 or hour > 22:
            temporal_risk += 0.3
        
        # Weekend transactions (if unusual for user)
        if current_time.weekday() >= 5:  # Weekend
            temporal_risk += 0.1
        
        # Session frequency (too many sessions in short time)
        if 'recent_session_count' in session_context:
            recent_sessions = session_context['recent_session_count']
            if recent_sessions > 5:  # More than 5 sessions in recent period
                temporal_risk += min(recent_sessions * 0.05, 0.4)
        
        # Time since last session
        if 'last_session_time' in session_context:
            last_session = session_context['last_session_time']
            if isinstance(last_session, datetime):
                time_diff = current_time - last_session
                if time_diff < timedelta(minutes=5):  # Very quick succession
                    temporal_risk += 0.2
        
        # Transaction velocity
        if 'transaction_velocity' in session_context:
            velocity = session_context['transaction_velocity']
            if velocity > 10:  # More than 10 transactions per hour
                temporal_risk += 0.3
        
        return min(temporal_risk, 1.0)
    
    def _calculate_device_risk(self, session_context: Dict) -> float:
        """Calculate risk based on device-related factors."""
        device_risk = 0.0
        
        # New or unrecognized device
        if session_context.get('is_new_device', False):
            device_risk += 0.5
        
        # Device fingerprint mismatch
        if session_context.get('device_fingerprint_match', 1.0) < 0.8:
            mismatch_severity = 1.0 - session_context['device_fingerprint_match']
            device_risk += mismatch_severity * 0.4
        
        # Rooted/jailbroken device
        if session_context.get('is_rooted_device', False):
            device_risk += 0.6
        
        # Unusual screen resolution or device specs
        if session_context.get('device_specs_anomaly', False):
            device_risk += 0.2
        
        # Multiple devices in short time
        if 'device_count_recent' in session_context:
            device_count = session_context['device_count_recent']
            if device_count > 2:
                device_risk += min(device_count * 0.1, 0.3)
        
        # Emulator detection
        if session_context.get('is_emulator', False):
            device_risk += 0.7
        
        return min(device_risk, 1.0)
    
    def _calculate_contextual_risk(self, session_context: Dict) -> float:
        """Calculate risk based on contextual factors."""
        contextual_risk = 0.0
        
        # Geographic location risk
        if session_context.get('unusual_location', False):
            contextual_risk += 0.4
        
        # Distance from usual location
        if 'location_distance_km' in session_context:
            distance = session_context['location_distance_km']
            if distance > 100:  # More than 100km from usual location
                contextual_risk += min(distance / 1000, 0.5)  # Scale with distance
        
        # VPN or proxy usage
        if session_context.get('using_vpn', False):
            contextual_risk += 0.3
        
        # Network type risk (public wifi, etc.)
        network_risk = session_context.get('network_risk_score', 0.0)
        contextual_risk += network_risk * 0.3
        
        # Transaction amount (if applicable)
        if 'transaction_amount' in session_context:
            amount = session_context['transaction_amount']
            avg_amount = session_context.get('user_avg_transaction', 1000)
            if amount > avg_amount * 3:  # 3x larger than average
                contextual_risk += 0.4
        
        # Failed authentication attempts
        if 'failed_auth_attempts' in session_context:
            failed_attempts = session_context['failed_auth_attempts']
            contextual_risk += min(failed_attempts * 0.2, 0.6)
        
        return min(contextual_risk, 1.0)
    
    def _calculate_historical_risk(self, user_id: Optional[str], 
                                 session_context: Dict) -> float:
        """Calculate risk based on historical patterns."""
        if not user_id or user_id not in self.user_risk_history:
            return 0.3  # Medium risk for new users
        
        user_history = self.user_risk_history[user_id]
        historical_risk = 0.0
        
        # Average historical risk
        if 'risk_scores' in user_history:
            recent_scores = user_history['risk_scores'][-10:]  # Last 10 sessions
            avg_risk = np.mean(recent_scores) if recent_scores else 0.3
            historical_risk += avg_risk * 0.5
        
        # Risk trend (increasing/decreasing)
        if 'risk_scores' in user_history and len(user_history['risk_scores']) >= 5:
            recent_trend = np.polyfit(
                range(len(user_history['risk_scores'][-5:])), 
                user_history['risk_scores'][-5:], 
                1
            )[0]  # Slope of trend line
            
            if recent_trend > 0.1:  # Increasing risk trend
                historical_risk += 0.3
        
        # Previous fraud incidents
        fraud_count = user_history.get('fraud_incidents', 0)
        if fraud_count > 0:
            historical_risk += min(fraud_count * 0.2, 0.5)
        
        # Account age (newer accounts are riskier)
        if 'account_age_days' in session_context:
            age_days = session_context['account_age_days']
            if age_days < 30:  # Account less than 30 days old
                historical_risk += 0.2
            elif age_days < 7:  # Account less than 7 days old
                historical_risk += 0.4
        
        return min(historical_risk, 1.0)
    
    def _calculate_risk_multiplier(self, session_context: Dict) -> float:
        """Calculate risk multiplier based on session context."""
        multiplier = 1.0
        
        for scenario, mult in self.risk_multipliers.items():
            if session_context.get(scenario, False):
                multiplier *= mult
        
        # Cap the multiplier to prevent extreme scores
        return min(multiplier, 3.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score and thresholds."""
        if risk_score >= self.thresholds['critical']:
            return RiskLevel.CRITICAL
        elif risk_score >= self.thresholds['high']:
            return RiskLevel.HIGH
        elif risk_score >= self.thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_risk_explanation(self, risk_components: Dict, 
                                 multiplier: float, session_context: Dict) -> List[str]:
        """Generate human-readable explanation of risk factors."""
        explanations = []
        
        # High-contributing risk factors
        for component, score in risk_components.items():
            if score > 0.3:
                component_name = component.replace('_risk', '').replace('_', ' ').title()
                explanations.append(f"Elevated {component_name} (score: {score:.2f})")
        
        # Risk multipliers
        if multiplier > 1.2:
            active_multipliers = [
                scenario.replace('_', ' ').title() 
                for scenario, active in session_context.items() 
                if active and scenario in self.risk_multipliers
            ]
            if active_multipliers:
                explanations.append(f"Risk amplified by: {', '.join(active_multipliers)}")
        
        if not explanations:
            explanations.append("No significant risk factors detected")
        
        return explanations
    
    def _get_recommended_actions(self, risk_level: RiskLevel) -> List[str]:
        """Get recommended actions based on risk level."""
        actions = {
            RiskLevel.LOW: [
                "Continue normal session monitoring",
                "Log interaction for future analysis"
            ],
            RiskLevel.MEDIUM: [
                "Increase monitoring frequency",
                "Consider additional verification for sensitive operations",
                "Log detailed session data"
            ],
            RiskLevel.HIGH: [
                "Require additional authentication",
                "Limit access to sensitive features",
                "Alert security team",
                "Detailed audit logging"
            ],
            RiskLevel.CRITICAL: [
                "Terminate session immediately",
                "Require multi-factor authentication",
                "Lock account temporarily",
                "Immediate security team notification",
                "Forensic data collection"
            ]
        }
        
        return actions.get(risk_level, ["No specific actions defined"])
    
    def _calculate_confidence(self, risk_components: Dict) -> float:
        """Calculate confidence in risk assessment."""
        # Higher confidence when multiple components agree on risk level
        risk_values = list(risk_components.values())
        
        if not risk_values:
            return 0.5
        
        # Calculate variance in risk components
        variance = np.var(risk_values)
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + variance * 5)
        
        return min(max(confidence, 0.1), 0.95)
    
    def _update_risk_history(self, user_id: str, risk_assessment: Dict) -> None:
        """Update user's risk history with new assessment."""
        if user_id not in self.user_risk_history:
            self.user_risk_history[user_id] = {
                'risk_scores': [],
                'fraud_incidents': 0,
                'last_updated': datetime.now()
            }
        
        user_history = self.user_risk_history[user_id]
        user_history['risk_scores'].append(risk_assessment['risk_score'])
        user_history['last_updated'] = datetime.now()
        
        # Keep only last 100 risk scores
        if len(user_history['risk_scores']) > 100:
            user_history['risk_scores'] = user_history['risk_scores'][-100:]
        
        # Update fraud incidents if high risk
        if risk_assessment['risk_level'] in ['high', 'critical']:
            user_history['fraud_incidents'] += 1
    
    def update_thresholds(self, low: float = None, medium: float = None, 
                         high: float = None, critical: float = None) -> None:
        """
        Update risk level thresholds.
        
        Args:
            low: Low risk threshold
            medium: Medium risk threshold
            high: High risk threshold
            critical: Critical risk threshold
            
        Raises:
            ValueError: If thresholds are invalid or not in ascending order
        """
        new_thresholds = self.thresholds.copy()
        
        if low is not None:
            if not 0 <= low <= 1:
                raise ValueError("Low threshold must be between 0 and 1")
            new_thresholds['low'] = low
        
        if medium is not None:
            if not 0 <= medium <= 1:
                raise ValueError("Medium threshold must be between 0 and 1")
            new_thresholds['medium'] = medium
        
        if high is not None:
            if not 0 <= high <= 1:
                raise ValueError("High threshold must be between 0 and 1")
            new_thresholds['high'] = high
        
        if critical is not None:
            if not 0 <= critical <= 1:
                raise ValueError("Critical threshold must be between 0 and 1")
            new_thresholds['critical'] = critical
        
        # Validate thresholds are in ascending order
        if not (new_thresholds['low'] < new_thresholds['medium'] < 
                new_thresholds['high'] < new_thresholds['critical']):
            raise ValueError("Risk thresholds must be in ascending order: low < medium < high < critical")
        
        self.thresholds = new_thresholds
    
    def get_user_risk_summary(self, user_id: str) -> Dict:
        """
        Get risk summary for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user risk summary
            
        Raises:
            ValueError: If user_id is not found in history
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        
        if user_id not in self.user_risk_history:
            raise ValueError(f"No risk history found for user {user_id}")
        
        user_history = self.user_risk_history[user_id]
        risk_scores = user_history['risk_scores']
        
        if not risk_scores:
            return {
                'user_id': user_id,
                'total_sessions': 0,
                'average_risk': 0.0,
                'risk_trend': 'insufficient_data',
                'fraud_incidents': user_history.get('fraud_incidents', 0),
                'last_updated': user_history['last_updated'],
                'risk_distribution': {
                    'low': 0,
                    'medium': 0,
                    'high': 0
                }
            }
        
        summary = {
            'user_id': user_id,
            'total_sessions': len(risk_scores),
            'average_risk': float(np.mean(risk_scores)),  # Convert to native float
            'risk_trend': 'stable',
            'fraud_incidents': user_history.get('fraud_incidents', 0),
            'last_updated': user_history['last_updated'],
            'risk_distribution': {
                'low': sum(1 for score in risk_scores if score < self.thresholds['medium']),
                'medium': sum(1 for score in risk_scores if self.thresholds['medium'] <= score < self.thresholds['high']),
                'high': sum(1 for score in risk_scores if score >= self.thresholds['high'])
            }
        }
        
        # Calculate risk trend
        if len(risk_scores) >= 5:
            recent_avg = np.mean(risk_scores[-5:])
            older_avg = np.mean(risk_scores[-10:-5]) if len(risk_scores) >= 10 else np.mean(risk_scores[:-5])
            
            if recent_avg > older_avg + 0.1:
                summary['risk_trend'] = 'increasing'
            elif recent_avg < older_avg - 0.1:
                summary['risk_trend'] = 'decreasing'
        else:
            summary['risk_trend'] = 'insufficient_data'
        
        return summary
