import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set environment variables to prevent TensorFlow threading issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Autoencoder model will be disabled.")

class AnomalyDetector:
    """
    Unsupervised anomaly detection for ransomware detection.
    Supports Isolation Forest and Autoencoder models.
    """
    
    def __init__(self, model_type='isolation_forest', contamination=0.03):
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.feature_cols = None
        
    def prepare_features(self, df):
        """Extract numerical features for modeling"""
        exclude_cols = ['user_id', 'user_id_original', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in [np.float64, np.int64]]
        self.feature_cols = feature_cols
        return df[feature_cols].values
    
    def train_isolation_forest(self, X):
        """Train Isolation Forest model"""
        print(f"Training Isolation Forest (contamination={self.contamination})...")
        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=1,  # Changed from -1 to avoid threading issues
            verbose=0
        )
        
        print("Fitting model... (this may take a moment)")
        self.model.fit(X)
        print("✓ Isolation Forest training complete")
    
    def train_autoencoder(self, X):
        """Train Autoencoder model for anomaly detection"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder model")
        
        print(f"Training Autoencoder...")
        
        input_dim = X.shape[1]
        encoding_dim = max(2, input_dim // 2)
        
        # Build autoencoder
        encoder = keras.Sequential([
            layers.Dense(encoding_dim * 2, activation='relu', input_shape=(input_dim,)),
            layers.Dense(encoding_dim, activation='relu')
        ])
        
        decoder = keras.Sequential([
            layers.Dense(encoding_dim * 2, activation='relu', input_shape=(encoding_dim,)),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        
        self.model = keras.Sequential([encoder, decoder])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.model.fit(
            X, X,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        print("✓ Autoencoder training complete")
    
    def train(self, df):
        """Train the anomaly detection model"""
        X = self.prepare_features(df)
        
        if self.model_type == 'isolation_forest':
            self.train_isolation_forest(X)
        elif self.model_type == 'autoencoder':
            self.train_autoencoder(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def predict(self, df):
        """Predict anomaly scores and labels"""
        X = self.prepare_features(df)
        
        if self.model_type == 'isolation_forest':
            # Get anomaly scores (lower = more anomalous)
            anomaly_scores = -self.model.score_samples(X)
            predictions = self.model.predict(X)
            # Convert to binary: -1 (anomaly) → 1, 1 (normal) → 0
            is_anomaly = (predictions == -1).astype(int)
            
        elif self.model_type == 'autoencoder':
            # Reconstruction error as anomaly score
            reconstructed = self.model.predict(X, verbose=0)
            anomaly_scores = np.mean(np.square(X - reconstructed), axis=1)
            
            # Threshold: top contamination% are anomalies
            threshold = np.percentile(anomaly_scores, (1 - self.contamination) * 100)
            is_anomaly = (anomaly_scores > threshold).astype(int)
        
        return anomaly_scores, is_anomaly
    
    def evaluate(self, df, true_labels=None):
        """Evaluate model performance if ground truth is available"""
        anomaly_scores, predictions = self.predict(df)
        
        results = {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'n_anomalies': predictions.sum(),
            'anomaly_rate': predictions.mean()
        }
        
        if true_labels is not None:
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            cm = confusion_matrix(true_labels, predictions)
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm
            })
            
            print("\n" + "="*50)
            print("MODEL EVALUATION")
            print("="*50)
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            print(f"Anomalies detected: {predictions.sum()} / {len(predictions)}")
            print("="*50)
        
        return results
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model_type == 'isolation_forest':
            joblib.dump(self.model, filepath)
        elif self.model_type == 'autoencoder':
            self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        if self.model_type == 'isolation_forest':
            self.model = joblib.load(filepath)
        elif self.model_type == 'autoencoder':
            self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")

def create_ground_truth_labels(df):
    """
    Create approximate ground truth labels based on extreme patterns.
    This simulates having ~1-2% labeled data for evaluation.
    """
    labels = np.zeros(len(df))
    
    # Define anomaly conditions (extreme patterns)
    conditions = (
        (df['file_encryption_rate'] > 10) |
        ((df['failed_logins'] > 8) & (df['session_duration'] < 500)) |
        ((df['data_outbound_mb'] > 400) & (df['hour_of_day'].isin([0,1,2,3,22,23]))) |
        ((df['files_accessed'] > 500) & (df['cpu_usage'] > 70))
    )
    
    labels[conditions] = 1
    return labels

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    df_scaled = pd.read_csv("banking_logs_scaled.csv")
    df_clean = pd.read_csv("banking_logs_clean.csv")
    
    # Create ground truth for evaluation
    print("Creating ground truth labels...")
    true_labels = create_ground_truth_labels(df_clean)
    print(f"Ground truth: {true_labels.sum()} anomalies ({true_labels.mean()*100:.1f}%)")
    
    # Train Isolation Forest
    print("\n" + "="*50)
    print("TRAINING ISOLATION FOREST MODEL")
    print("="*50)
    detector = AnomalyDetector(model_type='isolation_forest', contamination=0.03)
    detector.train(df_scaled)
    
    # Evaluate
    results = detector.evaluate(df_scaled, true_labels)
    
    # Add anomaly scores to dataframe
    df_clean['anomaly_score'] = results['anomaly_scores']
    df_clean['is_anomaly'] = results['predictions']
    
    # Save results
    df_clean.to_csv("banking_logs_with_scores.csv", index=False)
    detector.save_model("isolation_forest_model.pkl")
    
    print("\n✓ Results saved to banking_logs_with_scores.csv")
    print("✓ Model saved to isolation_forest_model.pkl")
    
    # Display top anomalies
    print("\n" + "="*50)
    print("TOP 10 ANOMALOUS SESSIONS")
    print("="*50)
    top_anomalies = df_clean.nlargest(10, 'anomaly_score')[
        ['user_id_original', 'files_accessed', 'failed_logins', 
         'data_outbound_mb', 'file_encryption_rate', 'anomaly_score']
    ]
    print(top_anomalies.to_string(index=False))