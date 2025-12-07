"""
Multi-Model Ensemble for Ransomware Detection
Combines multiple unsupervised models for improved detection accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class EnsembleAnomalyDetector:
    """
    Multi-model ensemble for anomaly detection
    Combines predictions from multiple unsupervised models
    """
    
    def __init__(self, contamination=0.03, voting='soft', weights=None):
        """
        Initialize ensemble detector
        
        Args:
            contamination: Expected proportion of outliers
            voting: 'soft' (average scores) or 'hard' (majority vote)
            weights: List of weights for each model (None for equal weights)
        """
        self.contamination = contamination
        self.voting = voting
        self.weights = weights
        self.scaler = StandardScaler()
        self.models = {}
        self.model_names = []
        
    def _initialize_models(self):
        """Initialize all base models"""
        models = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            ),
            'lof': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20,
                n_jobs=-1
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            ),
            'one_class_svm': OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        }
        return models
    
    def fit(self, X, model_subset=None):
        """
        Train all models in the ensemble
        
        Args:
            X: Training data (numpy array or DataFrame)
            model_subset: List of model names to use (None for all)
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models
        all_models = self._initialize_models()
        
        # Select model subset
        if model_subset:
            self.models = {k: v for k, v in all_models.items() if k in model_subset}
        else:
            self.models = all_models
        
        self.model_names = list(self.models.keys())
        
        # Set default weights if not provided
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        
        # Train each model
        print(f"Training {len(self.models)} models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_scaled)
        
        print("✅ Ensemble training complete!")
        return self
    
    def predict(self, X):
        """
        Predict anomalies using ensemble voting
        
        Returns:
            predictions: Binary predictions (1=anomaly, 0=normal)
            scores: Anomaly scores
            individual_predictions: Dict of predictions from each model
        """
        X_scaled = self.scaler.transform(X)
        
        individual_predictions = {}
        individual_scores = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            # Convert to binary (1=anomaly, 0=normal)
            pred_binary = (pred == -1).astype(int)
            individual_predictions[name] = pred_binary
            
            # Get anomaly scores (higher = more anomalous)
            if hasattr(model, 'score_samples'):
                scores = -model.score_samples(X_scaled)
            elif hasattr(model, 'decision_function'):
                scores = -model.decision_function(X_scaled)
            else:
                scores = pred_binary.astype(float)
            
            individual_scores[name] = scores
        
        # Ensemble voting
        if self.voting == 'soft':
            # Average weighted scores
            weighted_scores = np.zeros(len(X))
            for i, name in enumerate(self.model_names):
                weighted_scores += self.weights[i] * individual_scores[name]
            weighted_scores /= sum(self.weights)
            
            # Threshold at contamination level
            threshold = np.percentile(weighted_scores, (1 - self.contamination) * 100)
            ensemble_predictions = (weighted_scores >= threshold).astype(int)
            
        else:  # hard voting
            # Majority vote with weights
            votes = np.zeros(len(X))
            for i, name in enumerate(self.model_names):
                votes += self.weights[i] * individual_predictions[name]
            
            threshold = sum(self.weights) / 2
            ensemble_predictions = (votes > threshold).astype(int)
            weighted_scores = votes / sum(self.weights)
        
        return ensemble_predictions, weighted_scores, individual_predictions, individual_scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate ensemble performance
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            Dict with performance metrics
        """
        predictions, scores, ind_pred, ind_scores = self.predict(X)
        
        # Ensemble metrics
        ensemble_metrics = {
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'anomalies_detected': predictions.sum()
        }
        
        # Individual model metrics
        individual_metrics = {}
        for name in self.model_names:
            individual_metrics[name] = {
                'precision': precision_score(y_true, ind_pred[name], zero_division=0),
                'recall': recall_score(y_true, ind_pred[name], zero_division=0),
                'f1_score': f1_score(y_true, ind_pred[name], zero_division=0)
            }
        
        return {
            'ensemble': ensemble_metrics,
            'individual_models': individual_metrics
        }
    
    def save(self, filepath):
        """Save ensemble model to disk"""
        joblib.dump(self, filepath)
        print(f"✅ Ensemble saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load ensemble model from disk"""
        return joblib.load(filepath)


class StackedEnsemble:
    """
    Stacked ensemble using base models + meta-learner
    More sophisticated than simple voting
    """
    
    def __init__(self, base_models, meta_learner=None, contamination=0.03):
        """
        Args:
            base_models: Dict of base anomaly detectors
            meta_learner: Supervised classifier for final prediction
            contamination: Expected outlier proportion
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        if meta_learner is None:
            from sklearn.ensemble import RandomForestClassifier
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(self, X, y_true=None):
        """
        Train base models and meta-learner
        
        Args:
            X: Training features
            y_true: True labels (optional, for meta-learner)
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base models
        print("Training base models...")
        for name, model in self.base_models.items():
            print(f"  Training {name}...")
            model.fit(X_scaled)
        
        # If we have labels, train meta-learner
        if y_true is not None:
            print("Training meta-learner...")
            meta_features = self._get_meta_features(X_scaled)
            self.meta_learner.fit(meta_features, y_true)
            print("✅ Stacked ensemble complete!")
        
        return self
    
    def _get_meta_features(self, X_scaled):
        """Extract predictions from base models as meta-features"""
        meta_features = []
        
        for name, model in self.base_models.items():
            # Get scores
            if hasattr(model, 'score_samples'):
                scores = -model.score_samples(X_scaled)
            elif hasattr(model, 'decision_function'):
                scores = -model.decision_function(X_scaled)
            else:
                pred = model.predict(X_scaled)
                scores = (pred == -1).astype(float)
            
            meta_features.append(scores)
        
        return np.column_stack(meta_features)
    
    def predict(self, X):
        """Predict using stacked ensemble"""
        X_scaled = self.scaler.transform(X)
        meta_features = self._get_meta_features(X_scaled)
        
        # Use meta-learner for final prediction
        predictions = self.meta_learner.predict(meta_features)
        
        # Get prediction probabilities as scores
        if hasattr(self.meta_learner, 'predict_proba'):
            scores = self.meta_learner.predict_proba(meta_features)[:, 1]
        else:
            scores = predictions.astype(float)
        
        return predictions, scores


def compare_ensemble_strategies(X, y_true, contamination=0.03):
    """
    Compare different ensemble strategies
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # Strategy 1: Simple voting ensemble
    print("\n1. Testing Simple Voting Ensemble...")
    ensemble_voting = EnsembleAnomalyDetector(
        contamination=contamination,
        voting='soft'
    )
    ensemble_voting.fit(X)
    metrics_voting = ensemble_voting.evaluate(X, y_true)
    
    results.append({
        'strategy': 'Voting Ensemble',
        'precision': metrics_voting['ensemble']['precision'],
        'recall': metrics_voting['ensemble']['recall'],
        'f1_score': metrics_voting['ensemble']['f1_score']
    })
    
    # Strategy 2: Weighted ensemble (boost Isolation Forest)
    print("\n2. Testing Weighted Ensemble...")
    ensemble_weighted = EnsembleAnomalyDetector(
        contamination=contamination,
        voting='soft',
        weights=[2.0, 1.0, 1.0, 1.0]  # Double weight for IF
    )
    ensemble_weighted.fit(X)
    metrics_weighted = ensemble_weighted.evaluate(X, y_true)
    
    results.append({
        'strategy': 'Weighted Ensemble (IF x2)',
        'precision': metrics_weighted['ensemble']['precision'],
        'recall': metrics_weighted['ensemble']['recall'],
        'f1_score': metrics_weighted['ensemble']['f1_score']
    })
    
    # Strategy 3: Best 2 models only
    print("\n3. Testing Best-2 Ensemble...")
    ensemble_best2 = EnsembleAnomalyDetector(
        contamination=contamination,
        voting='soft'
    )
    ensemble_best2.fit(X, model_subset=['isolation_forest', 'lof'])
    metrics_best2 = ensemble_best2.evaluate(X, y_true)
    
    results.append({
        'strategy': 'Best-2 (IF + LOF)',
        'precision': metrics_best2['ensemble']['precision'],
        'recall': metrics_best2['ensemble']['recall'],
        'f1_score': metrics_best2['ensemble']['f1_score']
    })
    
    return pd.DataFrame(results)