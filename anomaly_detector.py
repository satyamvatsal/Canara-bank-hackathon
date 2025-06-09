import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Detects anomalies in user behavior using multiple machine learning algorithms.
    Supports both Isolation Forest and One-Class SVM for robust anomaly detection.
    """
    
    def __init__(self):
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.trained_models = {}
        self.training_scores = {}
        self.contamination_rate = 0.1
        self.is_trained_isolation = False
        self.is_trained_svm = False
    
    def train_isolation_forest(self, training_data: pd.DataFrame, 
                             contamination: float = 0.1, 
                             n_estimators: int = 100,
                             random_state: int = 42) -> Dict:
        """
        Train Isolation Forest model for anomaly detection.
        
        Args:
            training_data: DataFrame containing training data
            contamination: Expected proportion of anomalies in dataset
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing training results
            
        Raises:
            ValueError: If training data is empty or invalid
        """
        if training_data.empty:
            raise ValueError("Training data cannot be empty")
        
        if not (0 < contamination < 1):
            raise ValueError("Contamination rate must be between 0 and 1")
        
        # Prepare features
        feature_columns = [col for col in training_data.columns 
                         if col not in ['user_id', 'session_id', 'timestamp', 'is_anomaly']]
        
        if not feature_columns:
            raise ValueError("No valid feature columns found in training data")
        
        self.feature_columns = feature_columns
        X = training_data[feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.isolation_forest.fit(X_scaled)
        self.contamination_rate = contamination
        self.is_trained_isolation = True
        
        # Calculate training scores
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        
        training_results = {
            'model_type': 'Isolation Forest',
            'n_samples': len(training_data),
            'n_features': len(feature_columns),
            'contamination_rate': contamination,
            'n_estimators': n_estimators,
            'anomaly_threshold': np.percentile(anomaly_scores, contamination * 100),
            'training_anomalies': (predictions == -1).sum(),
            'training_accuracy': (predictions == 1).sum() / len(predictions)
        }
        
        self.training_scores['isolation_forest'] = {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'threshold': training_results['anomaly_threshold']
        }
        
        self.trained_models['isolation_forest'] = training_results
        
        return training_results
    
    def train_svm(self, training_data: pd.DataFrame,
                  nu: float = 0.1,
                  kernel: str = 'rbf',
                  gamma: str = 'scale') -> Dict:
        """
        Train One-Class SVM model for anomaly detection.
        
        Args:
            training_data: DataFrame containing training data
            nu: Upper bound on the fraction of training errors
            kernel: Kernel type for SVM
            gamma: Kernel coefficient
            
        Returns:
            Dictionary containing training results
            
        Raises:
            ValueError: If training data is empty or invalid
        """
        if training_data.empty:
            raise ValueError("Training data cannot be empty")
        
        if not (0 < nu < 1):
            raise ValueError("Nu parameter must be between 0 and 1")
        
        # Use same feature columns as Isolation Forest if available
        if not self.feature_columns:
            feature_columns = [col for col in training_data.columns 
                             if col not in ['user_id', 'session_id', 'timestamp', 'is_anomaly']]
            if not feature_columns:
                raise ValueError("No valid feature columns found in training data")
            self.feature_columns = feature_columns
        
        X = training_data[self.feature_columns].fillna(0)
        
        # Use existing scaler or fit new one
        if not hasattr(self.scaler, 'scale_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Train One-Class SVM
        self.one_class_svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        
        self.one_class_svm.fit(X_scaled)
        self.is_trained_svm = True
        
        # Calculate training scores
        predictions = self.one_class_svm.predict(X_scaled)
        decision_scores = self.one_class_svm.decision_function(X_scaled)
        
        training_results = {
            'model_type': 'One-Class SVM',
            'n_samples': len(training_data),
            'n_features': len(self.feature_columns),
            'nu': nu,
            'kernel': kernel,
            'gamma': gamma,
            'training_anomalies': (predictions == -1).sum(),
            'training_accuracy': (predictions == 1).sum() / len(predictions)
        }
        
        self.training_scores['one_class_svm'] = {
            'decision_scores': decision_scores,
            'predictions': predictions,
            'threshold': 0.0  # SVM uses 0 as threshold
        }
        
        self.trained_models['one_class_svm'] = training_results
        
        return training_results
    
    def detect_anomaly(self, session_data: pd.DataFrame, 
                      use_ensemble: bool = True) -> Dict:
        """
        Detect anomalies in session data using trained models.
        
        Args:
            session_data: DataFrame containing session behavioral features
            use_ensemble: Whether to use ensemble of models for prediction
            
        Returns:
            Dictionary containing anomaly detection results
            
        Raises:
            ValueError: If no models are trained or if features are incompatible
        """
        if not (self.is_trained_isolation or self.is_trained_svm):
            raise ValueError("No trained models available for anomaly detection")
        
        if session_data.empty:
            raise ValueError("Session data cannot be empty")
        
        # Validate features
        missing_features = [f for f in self.feature_columns if f not in session_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare features
        X = session_data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        results = {
            'timestamp': datetime.now(),
            'n_samples': len(session_data),
            'model_predictions': {},
            'ensemble_prediction': None,
            'anomaly_scores': {},
            'confidence_scores': {}
        }
        
        # Isolation Forest prediction
        if self.is_trained_isolation:
            if_predictions = self.isolation_forest.predict(X_scaled)
            if_scores = self.isolation_forest.decision_function(X_scaled)
            
            results['model_predictions']['isolation_forest'] = if_predictions
            results['anomaly_scores']['isolation_forest'] = if_scores
            results['confidence_scores']['isolation_forest'] = self._calculate_confidence(
                if_scores, self.training_scores['isolation_forest']['threshold']
            )
        
        # One-Class SVM prediction
        if self.is_trained_svm:
            svm_predictions = self.one_class_svm.predict(X_scaled)
            svm_scores = self.one_class_svm.decision_function(X_scaled)
            
            results['model_predictions']['one_class_svm'] = svm_predictions
            results['anomaly_scores']['one_class_svm'] = svm_scores
            results['confidence_scores']['one_class_svm'] = self._calculate_confidence(
                svm_scores, 0.0
            )
        
        # Ensemble prediction
        if use_ensemble and len(results['model_predictions']) > 1:
            ensemble_pred = self._ensemble_prediction(results['model_predictions'])
            results['ensemble_prediction'] = ensemble_pred
        elif len(results['model_predictions']) == 1:
            # Use single model prediction
            model_name = list(results['model_predictions'].keys())[0]
            results['ensemble_prediction'] = results['model_predictions'][model_name]
        
        # Calculate overall anomaly probability
        results['anomaly_probability'] = self._calculate_anomaly_probability(results)
        
        return results
    
    def _calculate_confidence(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Calculate confidence scores based on distance from decision boundary."""
        distances = np.abs(scores - threshold)
        # Normalize distances to [0, 1] range
        max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
        confidence = distances / max_distance
        return np.clip(confidence, 0.0, 1.0)
    
    def _ensemble_prediction(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models using voting."""
        predictions = list(model_predictions.values())
        if len(predictions) == 0:
            return np.array([])
        
        # Stack predictions and use majority voting
        stacked_predictions = np.column_stack(predictions)
        ensemble_pred = np.array([
            1 if np.mean(row) > 0 else -1 
            for row in stacked_predictions
        ])
        
        return ensemble_pred
    
    def _calculate_anomaly_probability(self, results: Dict) -> np.ndarray:
        """Calculate probability of anomaly based on all available scores."""
        if not results['anomaly_scores']:
            return np.array([0.5] * results['n_samples'])
        
        probabilities = []
        
        for model_name, scores in results['anomaly_scores'].items():
            if model_name == 'isolation_forest':
                # Convert isolation forest scores to probabilities
                threshold = self.training_scores['isolation_forest']['threshold']
                # Handle extreme values to prevent overflow
                adjusted_scores = np.clip(scores - threshold, -10, 10)
                prob = 1 / (1 + np.exp(5 * adjusted_scores))  # Sigmoid transformation
            else:  # one_class_svm
                # Convert SVM scores to probabilities
                # Handle extreme values to prevent overflow
                adjusted_scores = np.clip(scores, -10, 10)
                prob = 1 / (1 + np.exp(2 * adjusted_scores))  # Sigmoid transformation
            
            probabilities.append(prob)
        
        if probabilities:
            # Average probabilities from all models
            return np.mean(probabilities, axis=0)
        else:
            return np.array([0.5] * results['n_samples'])
    
    def evaluate_models(self, test_data: pd.DataFrame, 
                       true_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate trained models on test data.
        
        Args:
            test_data: Test dataset
            true_labels: True anomaly labels (1 for normal, -1 for anomaly)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not (self.is_trained_isolation or self.is_trained_svm):
            return {'error': 'No trained models to evaluate'}
        
        X = test_data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        evaluation_results = {}
        
        # Evaluate Isolation Forest
        if self.is_trained_isolation:
            if_pred = self.isolation_forest.predict(X_scaled)
            if_scores = self.isolation_forest.decision_function(X_scaled)
            
            eval_metrics = {
                'predictions': if_pred,
                'anomaly_scores': if_scores,
                'anomaly_rate': (if_pred == -1).sum() / len(if_pred),
                'score_statistics': {
                    'mean': np.mean(if_scores),
                    'std': np.std(if_scores),
                    'min': np.min(if_scores),
                    'max': np.max(if_scores)
                }
            }
            
            if true_labels is not None:
                eval_metrics['classification_report'] = classification_report(
                    true_labels, if_pred, output_dict=True
                )
                eval_metrics['confusion_matrix'] = confusion_matrix(true_labels, if_pred)
            
            evaluation_results['isolation_forest'] = eval_metrics
        
        # Evaluate One-Class SVM
        if self.is_trained_svm:
            svm_pred = self.one_class_svm.predict(X_scaled)
            svm_scores = self.one_class_svm.decision_function(X_scaled)
            
            eval_metrics = {
                'predictions': svm_pred,
                'decision_scores': svm_scores,
                'anomaly_rate': (svm_pred == -1).sum() / len(svm_pred),
                'score_statistics': {
                    'mean': np.mean(svm_scores),
                    'std': np.std(svm_scores),
                    'min': np.min(svm_scores),
                    'max': np.max(svm_scores)
                }
            }
            
            if true_labels is not None:
                eval_metrics['classification_report'] = classification_report(
                    true_labels, svm_pred, output_dict=True
                )
                eval_metrics['confusion_matrix'] = confusion_matrix(true_labels, svm_pred)
            
            evaluation_results['one_class_svm'] = eval_metrics
        
        return evaluation_results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (approximated for unsupervised models).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.feature_columns:
            return {}
        
        importance_scores = {}
        
        # For Isolation Forest, we can use path lengths as importance proxy
        if self.is_trained_isolation:
            # Generate random samples to estimate feature importance
            n_samples = 1000
            random_data = np.random.randn(n_samples, len(self.feature_columns))
            
            baseline_scores = self.isolation_forest.decision_function(random_data)
            
            for i, feature in enumerate(self.feature_columns):
                # Permute feature and measure score change
                permuted_data = random_data.copy()
                np.random.shuffle(permuted_data[:, i])
                
                permuted_scores = self.isolation_forest.decision_function(permuted_data)
                importance = np.mean(np.abs(baseline_scores - permuted_scores))
                importance_scores[feature] = importance
        
        # Normalize importance scores
        if importance_scores:
            max_importance = max(importance_scores.values())
            if max_importance > 0:
                importance_scores = {
                    k: v / max_importance for k, v in importance_scores.items()
                }
        
        return importance_scores
    
    def is_trained(self) -> bool:
        """Check if at least one model is trained."""
        return self.is_trained_isolation or self.is_trained_svm
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models to disk."""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'one_class_svm': self.one_class_svm,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'trained_models': self.trained_models,
            'training_scores': self.training_scores,
            'contamination_rate': self.contamination_rate,
            'is_trained_isolation': self.is_trained_isolation,
            'is_trained_svm': self.is_trained_svm
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        
        self.isolation_forest = model_data['isolation_forest']
        self.one_class_svm = model_data['one_class_svm']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.trained_models = model_data['trained_models']
        self.training_scores = model_data['training_scores']
        self.contamination_rate = model_data['contamination_rate']
        self.is_trained_isolation = model_data['is_trained_isolation']
        self.is_trained_svm = model_data['is_trained_svm']
    
    def get_model_info(self) -> Dict:
        """Get information about trained models."""
        info = {
            'models_trained': [],
            'feature_count': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy(),
            'training_info': self.trained_models.copy()
        }
        
        if self.is_trained_isolation:
            info['models_trained'].append('Isolation Forest')
        
        if self.is_trained_svm:
            info['models_trained'].append('One-Class SVM')
        
        return info
