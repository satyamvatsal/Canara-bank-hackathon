import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BehaviorAnalyzer:
    """
    Analyzes user behavior patterns and creates behavioral profiles
    for continuous authentication in mobile banking applications.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.feature_importance = {}
        self.user_profiles = {}
        self.baseline_features = [
            'typing_speed_mean', 'typing_speed_std', 'typing_interval_mean',
            'touch_pressure_mean', 'touch_pressure_std', 'swipe_velocity_mean',
            'swipe_velocity_std', 'swipe_direction_consistency', 'tap_duration_mean',
            'tap_duration_std', 'scroll_speed_mean', 'scroll_speed_std',
            'device_orientation_changes', 'session_duration', 'navigation_pattern_score'
        ]
        self.is_trained = False
    
    def extract_behavioral_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral features from raw interaction data.
        
        Args:
            raw_data: DataFrame containing raw user interaction data
            
        Returns:
            DataFrame with extracted behavioral features
        """
        features = pd.DataFrame()
        
        # Group by user and session for feature extraction
        for (user_id, session_id), group in raw_data.groupby(['user_id', 'session_id']):
            session_features = self._extract_session_features(group)
            session_features['user_id'] = user_id
            session_features['session_id'] = session_id
            features = pd.concat([features, pd.DataFrame([session_features])], ignore_index=True)
        
        return features
    
    def _extract_session_features(self, session_data: pd.DataFrame) -> Dict:
        """Extract features from a single session."""
        features = {}
        
        # Typing patterns
        if 'keystroke_time' in session_data.columns:
            typing_intervals = session_data['keystroke_time'].diff().dropna()
            features['typing_speed_mean'] = 1 / typing_intervals.mean() if len(typing_intervals) > 0 else 0
            features['typing_speed_std'] = typing_intervals.std() if len(typing_intervals) > 0 else 0
            features['typing_interval_mean'] = typing_intervals.mean() if len(typing_intervals) > 0 else 0
        else:
            features.update({
                'typing_speed_mean': 0,
                'typing_speed_std': 0,
                'typing_interval_mean': 0
            })
        
        # Touch patterns
        if 'touch_pressure' in session_data.columns:
            features['touch_pressure_mean'] = session_data['touch_pressure'].mean()
            features['touch_pressure_std'] = session_data['touch_pressure'].std()
        else:
            features['touch_pressure_mean'] = 0.5
            features['touch_pressure_std'] = 0.1
        
        # Swipe patterns
        if 'swipe_velocity' in session_data.columns:
            features['swipe_velocity_mean'] = session_data['swipe_velocity'].mean()
            features['swipe_velocity_std'] = session_data['swipe_velocity'].std()
        else:
            features['swipe_velocity_mean'] = 100
            features['swipe_velocity_std'] = 20
        
        # Swipe direction consistency
        if 'swipe_direction' in session_data.columns:
            direction_changes = (session_data['swipe_direction'].diff() != 0).sum()
            features['swipe_direction_consistency'] = 1 - (direction_changes / len(session_data))
        else:
            features['swipe_direction_consistency'] = 0.8
        
        # Tap patterns
        if 'tap_duration' in session_data.columns:
            features['tap_duration_mean'] = session_data['tap_duration'].mean()
            features['tap_duration_std'] = session_data['tap_duration'].std()
        else:
            features['tap_duration_mean'] = 0.15
            features['tap_duration_std'] = 0.05
        
        # Scroll patterns
        if 'scroll_speed' in session_data.columns:
            features['scroll_speed_mean'] = session_data['scroll_speed'].mean()
            features['scroll_speed_std'] = session_data['scroll_speed'].std()
        else:
            features['scroll_speed_mean'] = 50
            features['scroll_speed_std'] = 15
        
        # Device handling
        features['device_orientation_changes'] = session_data.get('orientation_change', pd.Series([0])).sum()
        
        # Session characteristics
        if 'timestamp' in session_data.columns:
            session_duration = (session_data['timestamp'].max() - session_data['timestamp'].min()).total_seconds()
            features['session_duration'] = session_duration
        else:
            features['session_duration'] = 300  # Default 5 minutes
        
        # Navigation patterns
        if 'screen_transition' in session_data.columns:
            unique_screens = session_data['screen_transition'].nunique()
            total_transitions = len(session_data)
            features['navigation_pattern_score'] = unique_screens / total_transitions if total_transitions > 0 else 0
        else:
            features['navigation_pattern_score'] = 0.3
        
        return features
    
    def create_user_profile(self, user_id: str, user_data: pd.DataFrame) -> Dict:
        """
        Create a behavioral profile for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            user_data: Historical behavioral data for the user
            
        Returns:
            Dictionary containing the user's behavioral profile
        """
        profile = {
            'user_id': user_id,
            'profile_created': datetime.now(),
            'total_sessions': len(user_data),
            'behavioral_patterns': {},
            'stability_metrics': {},
            'risk_indicators': {}
        }
        
        # Calculate behavioral patterns (means and standard deviations)
        for feature in self.baseline_features:
            if feature in user_data.columns:
                profile['behavioral_patterns'][f'{feature}_baseline'] = user_data[feature].mean()
                profile['behavioral_patterns'][f'{feature}_variance'] = user_data[feature].std()
        
        # Calculate stability metrics
        profile['stability_metrics']['consistency_score'] = self._calculate_consistency_score(user_data)
        profile['stability_metrics']['temporal_stability'] = self._calculate_temporal_stability(user_data)
        
        # Calculate risk indicators
        profile['risk_indicators']['outlier_frequency'] = self._calculate_outlier_frequency(user_data)
        profile['risk_indicators']['pattern_deviation'] = self._calculate_pattern_deviation(user_data)
        
        # Store profile
        self.user_profiles[user_id] = profile
        
        return profile
    
    def _calculate_consistency_score(self, user_data: pd.DataFrame) -> float:
        """Calculate how consistent a user's behavior is over time."""
        if len(user_data) < 2:
            return 1.0
        
        consistency_scores = []
        for feature in self.baseline_features:
            if feature in user_data.columns and user_data[feature].std() > 0:
                cv = user_data[feature].std() / user_data[feature].mean()  # Coefficient of variation
                consistency_scores.append(1 / (1 + cv))  # Higher consistency = lower CV
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_temporal_stability(self, user_data: pd.DataFrame) -> float:
        """Calculate how stable behavior patterns are over time."""
        if len(user_data) < 5:
            return 1.0
        
        # Split data into windows and compare patterns
        window_size = max(5, len(user_data) // 4)
        windows = [user_data.iloc[i:i+window_size] for i in range(0, len(user_data)-window_size+1, window_size)]
        
        if len(windows) < 2:
            return 1.0
        
        stability_scores = []
        for feature in self.baseline_features:
            if feature in user_data.columns:
                window_means = [window[feature].mean() for window in windows if len(window) > 0]
                if len(window_means) > 1:
                    stability = 1 - (np.std(window_means) / (np.mean(window_means) + 1e-6))
                    stability_scores.append(max(0, stability))
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _calculate_outlier_frequency(self, user_data: pd.DataFrame) -> float:
        """Calculate how frequently the user exhibits outlier behavior."""
        outlier_counts = []
        
        for feature in self.baseline_features:
            if feature in user_data.columns and len(user_data) > 3:
                Q1 = user_data[feature].quantile(0.25)
                Q3 = user_data[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((user_data[feature] < (Q1 - 1.5 * IQR)) | 
                           (user_data[feature] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts.append(outliers / len(user_data))
        
        return np.mean(outlier_counts) if outlier_counts else 0.0
    
    def _calculate_pattern_deviation(self, user_data: pd.DataFrame) -> float:
        """Calculate deviation from typical behavioral patterns."""
        if len(user_data) < 5:
            return 0.0
        
        deviations = []
        for feature in self.baseline_features:
            if feature in user_data.columns:
                feature_mean = user_data[feature].mean()
                feature_std = user_data[feature].std()
                if feature_std > 0:
                    z_scores = np.abs((user_data[feature] - feature_mean) / feature_std)
                    avg_deviation = z_scores.mean()
                    deviations.append(min(avg_deviation / 3, 1.0))  # Normalize to 0-1
        
        return np.mean(deviations) if deviations else 0.0
    
    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the behavior analyzer on historical data.
        
        Args:
            training_data: DataFrame containing historical behavioral data
            
        Raises:
            ValueError: If training data is empty or missing required features
        """
        if training_data.empty:
            raise ValueError("Training data cannot be empty")
        
        # Validate required features exist
        missing_features = [f for f in self.baseline_features if f not in training_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Ensure we have the required features
        feature_data = training_data[self.baseline_features].fillna(0)
        
        # Fit the scaler
        self.scaler.fit(feature_data)
        scaled_data = self.scaler.transform(feature_data)
        
        # Fit PCA for dimensionality reduction
        self.pca.fit(scaled_data)
        
        # Fit KMeans for behavioral clustering
        pca_data = self.pca.transform(scaled_data)
        self.kmeans.fit(pca_data)
        
        # Calculate feature importance based on PCA components
        feature_importance = {}
        components = np.abs(self.pca.components_)
        explained_variance_ratio = self.pca.explained_variance_ratio_
        
        for i, feature in enumerate(self.baseline_features):
            # Weight component contributions by their explained variance
            importance = np.sum(components[:, i] * explained_variance_ratio)
            feature_importance[feature] = importance
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        self.feature_importance = {
            k: v/total_importance for k, v in feature_importance.items()
        }
        
        self.is_trained = True
    
    def analyze_session(self, session_data: pd.DataFrame, user_id: str) -> Dict:
        """
        Analyze a user session and compare it to their behavioral profile.
        
        Args:
            session_data: DataFrame containing session data
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.is_trained:
            raise ValueError("Behavior analyzer must be trained before analyzing sessions")
        
        if user_id not in self.user_profiles:
            raise ValueError(f"No behavioral profile found for user {user_id}")
        
        # Extract features from session
        session_features = self.extract_behavioral_features(session_data)
        
        # Get user profile
        user_profile = self.user_profiles[user_id]
        
        # Initialize analysis result
        analysis_result = {
            'user_id': user_id,
            'session_id': session_data.get('session_id', None),
            'timestamp': datetime.now(),
            'profile_match_score': 0.0,
            'behavioral_deviations': {},
            'risk_factors': [],
            'confidence_score': 0.0
        }
        
        # Compare session to profile
        analysis_result = self._compare_to_profile(
            session_features.iloc[0] if len(session_features) > 0 else pd.Series(),
            user_profile,
            analysis_result
        )
        
        return analysis_result
    
    def _compare_to_profile(self, session_features: pd.Series, 
                          user_profile: Dict, analysis_result: Dict) -> Dict:
        """Compare session features to user's behavioral profile."""
        feature_deviations = {}
        total_weighted_deviation = 0.0
        total_weight = 0.0
        
        for feature in self.baseline_features:
            if feature in session_features:
                baseline = user_profile['behavioral_patterns'].get(f'{feature}_baseline', 0)
                variance = user_profile['behavioral_patterns'].get(f'{feature}_variance', 1)
                
                if baseline != 0 or variance != 0:
                    # Calculate z-score deviation
                    deviation = abs(session_features[feature] - baseline) / (variance + 1e-6)
                    is_anomaly = deviation > 2.0  # More than 2 standard deviations
                    
                    feature_deviations[feature] = {
                        'value': float(session_features[feature]),
                        'baseline': float(baseline),
                        'deviation': float(deviation),
                        'is_anomaly': is_anomaly
                    }
                    
                    # Weight deviation by feature importance
                    weight = self.feature_importance.get(feature, 1.0)
                    total_weighted_deviation += deviation * weight
                    total_weight += weight
                    
                    if is_anomaly:
                        analysis_result['risk_factors'].append(
                            f"Anomalous {feature.replace('_', ' ')} pattern detected"
                        )
        
        # Calculate overall profile match score (inverse of weighted average deviation)
        if total_weight > 0:
            avg_weighted_deviation = total_weighted_deviation / total_weight
            profile_match_score = 1.0 / (1.0 + avg_weighted_deviation)
            analysis_result['profile_match_score'] = min(max(profile_match_score, 0.0), 1.0)
        
        analysis_result['behavioral_deviations'] = feature_deviations
        
        # Calculate confidence based on amount of available data
        n_features_available = sum(1 for f in self.baseline_features if f in session_features)
        analysis_result['confidence_score'] = n_features_available / len(self.baseline_features)
        
        return analysis_result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get the importance scores for each behavioral feature."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.feature_importance
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and profiles to disk."""
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'kmeans': self.kmeans,
            'feature_importance': self.feature_importance,
            'user_profiles': self.user_profiles,
            'baseline_features': self.baseline_features,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and profiles from disk."""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.kmeans = model_data['kmeans']
        self.feature_importance = model_data['feature_importance']
        self.user_profiles = model_data['user_profiles']
        self.baseline_features = model_data['baseline_features']
        self.is_trained = model_data['is_trained']
