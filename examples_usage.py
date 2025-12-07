"""
Usage Examples for Ensemble Ransomware Detection
Demonstrates various ways to use the ensemble models
"""

import numpy as np
import pandas as pd
from generate_data import generate_banking_data
from ensemble_models import EnsembleAnomalyDetector, StackedEnsemble, compare_ensemble_strategies
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# EXAMPLE 1: Basic Ensemble Usage
# ============================================================================
def example_basic_ensemble():
    """Simple ensemble training and prediction"""
    print("="*70)
    print("EXAMPLE 1: Basic Ensemble Usage")
    print("="*70)
    
    # Generate data
    print("\n1. Generating data...")
    df = generate_banking_data(5000)
    
    # Preprocess
    print("2. Preprocessing...")
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Prepare features
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    exclude_cols = ['user_id', 'timestamp', 'user_id_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols 
                   and df[col].dtype in [np.float64, np.int64]]
    X = df[feature_cols].values
    
    # Train ensemble
    print("3. Training ensemble...")
    ensemble = EnsembleAnomalyDetector(contamination=0.03)
    ensemble.fit(X)
    
    # Predict
    print("4. Making predictions...")
    predictions, scores, ind_preds, ind_scores = ensemble.predict(X)
    
    print(f"\n✅ Detected {predictions.sum()} anomalies out of {len(predictions)} samples")
    print(f"   ({predictions.mean()*100:.1f}% anomaly rate)")
    
    return ensemble, predictions, scores


# ============================================================================
# EXAMPLE 2: Weighted Ensemble
# ============================================================================
def example_weighted_ensemble():
    """Use custom weights to boost certain models"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Weighted Ensemble")
    print("="*70)
    
    # Generate data
    df = generate_banking_data(3000)
    df = df.drop_duplicates()
    
    # Quick preprocessing
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    feature_cols = ['files_accessed', 'failed_logins', 'data_outbound_mb', 
                   'hour_of_day', 'cpu_usage', 'file_encryption_rate', 
                   'session_duration', 'user_id_encoded']
    X = df[feature_cols].values
    
    # Create ground truth
    true_labels = np.zeros(len(df))
    conditions = (
        (df['file_encryption_rate'] > 10) |
        ((df['failed_logins'] > 8) & (df['session_duration'] < 500))
    )
    true_labels[conditions] = 1
    
    print("\n1. Training equal-weight ensemble...")
    ensemble_equal = EnsembleAnomalyDetector(contamination=0.03, weights=None)
    ensemble_equal.fit(X)
    metrics_equal = ensemble_equal.evaluate(X, true_labels)
    
    print(f"   Equal weights F1: {metrics_equal['ensemble']['f1_score']:.2%}")
    
    print("\n2. Training weighted ensemble (boost IF and LOF)...")
    ensemble_weighted = EnsembleAnomalyDetector(
        contamination=0.03,
        weights=[2.0, 1.5, 1.0, 1.0]  # IF, LOF, Elliptic, SVM
    )
    ensemble_weighted.fit(X)
    metrics_weighted = ensemble_weighted.evaluate(X, true_labels)
    
    print(f"   Weighted F1: {metrics_weighted['ensemble']['f1_score']:.2%}")
    
    improvement = (metrics_weighted['ensemble']['f1_score'] - 
                  metrics_equal['ensemble']['f1_score']) / \
                  metrics_equal['ensemble']['f1_score'] * 100
    
    print(f"\n🚀 Improvement with weighting: {improvement:+.1f}%")


# ============================================================================
# EXAMPLE 3: Model Subset Selection
# ============================================================================
def example_model_subset():
    """Use only selected models in the ensemble"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Model Subset Selection")
    print("="*70)
    
    # Generate data
    df = generate_banking_data(3000)
    df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    feature_cols = ['files_accessed', 'failed_logins', 'data_outbound_mb', 
                   'hour_of_day', 'cpu_usage', 'file_encryption_rate', 
                   'session_duration', 'user_id_encoded']
    X = df[feature_cols].values
    
    print("\n1. Training with all 4 models...")
    ensemble_all = EnsembleAnomalyDetector(contamination=0.03)
    ensemble_all.fit(X)
    preds_all, _, _, _ = ensemble_all.predict(X)
    
    print(f"   All models detected: {preds_all.sum()} anomalies")
    
    print("\n2. Training with best 2 models (IF + LOF)...")
    ensemble_best2 = EnsembleAnomalyDetector(contamination=0.03)
    ensemble_best2.fit(X, model_subset=['isolation_forest', 'lof'])
    preds_best2, _, _, _ = ensemble_best2.predict(X)
    
    print(f"   Best-2 detected: {preds_best2.sum()} anomalies")
    
    print("\n3. Training with fast models only (IF + Elliptic)...")
    ensemble_fast = EnsembleAnomalyDetector(contamination=0.03)
    ensemble_fast.fit(X, model_subset=['isolation_forest', 'elliptic_envelope'])
    preds_fast, _, _, _ = ensemble_fast.predict(X)
    
    print(f"   Fast models detected: {preds_fast.sum()} anomalies")


# ============================================================================
# EXAMPLE 4: Voting Strategy Comparison
# ============================================================================
def example_voting_strategies():
    """Compare soft vs hard voting"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Voting Strategy Comparison")
    print("="*70)
    
    # Generate data
    df = generate_banking_data(3000)
    df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    feature_cols = ['files_accessed', 'failed_logins', 'data_outbound_mb', 
                   'hour_of_day', 'cpu_usage', 'file_encryption_rate', 
                   'session_duration', 'user_id_encoded']
    X = df[feature_cols].values
    
    # Ground truth
    true_labels = np.zeros(len(df))
    conditions = (df['file_encryption_rate'] > 10) | \
                 ((df['failed_logins'] > 8) & (df['session_duration'] < 500))
    true_labels[conditions] = 1
    
    print("\n1. Soft voting (average scores)...")
    ensemble_soft = EnsembleAnomalyDetector(contamination=0.03, voting='soft')
    ensemble_soft.fit(X)
    metrics_soft = ensemble_soft.evaluate(X, true_labels)
    
    print(f"   Precision: {metrics_soft['ensemble']['precision']:.2%}")
    print(f"   Recall:    {metrics_soft['ensemble']['recall']:.2%}")
    print(f"   F1-Score:  {metrics_soft['ensemble']['f1_score']:.2%}")
    
    print("\n2. Hard voting (majority vote)...")
    ensemble_hard = EnsembleAnomalyDetector(contamination=0.03, voting='hard')
    ensemble_hard.fit(X)
    metrics_hard = ensemble_hard.evaluate(X, true_labels)
    
    print(f"   Precision: {metrics_hard['ensemble']['precision']:.2%}")
    print(f"   Recall:    {metrics_hard['ensemble']['recall']:.2%}")
    print(f"   F1-Score:  {metrics_hard['ensemble']['f1_score']:.2%}")


# ============================================================================
# EXAMPLE 5: Save and Load Model
# ============================================================================
def example_save_load():
    """Save trained model and load it later"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Save and Load Model")
    print("="*70)
    
    # Generate and prepare data
    df = generate_banking_data(2000)
    df = df.drop_duplicates()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    feature_cols = ['files_accessed', 'failed_logins', 'data_outbound_mb', 
                   'hour_of_day', 'cpu_usage', 'file_encryption_rate', 
                   'session_duration', 'user_id_encoded']
    X = df[feature_cols].values
    
    print("\n1. Training ensemble...")
    ensemble = EnsembleAnomalyDetector(contamination=0.03)
    ensemble.fit(X)
    
    print("2. Saving model to disk...")
    ensemble.save("my_ensemble_model.pkl")
    
    print("3. Loading model from disk...")
    loaded_ensemble = EnsembleAnomalyDetector.load("my_ensemble_model.pkl")
    
    print("4. Testing loaded model...")
    predictions, scores, _, _ = loaded_ensemble.predict(X[:10])
    
    print(f"\n✅ Successfully loaded and tested model")
    print(f"   Sample predictions: {predictions}")
    print(f"   Sample scores: {scores}")


# ============================================================================
# EXAMPLE 6: Real-time Prediction Loop
# ============================================================================
def example_realtime_prediction():
    """Simulate real-time anomaly detection"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Real-time Prediction Simulation")
    print("="*70)
    
    # Train ensemble once
    print("\n1. Training ensemble on historical data...")
    df_train = generate_banking_data(3000)
    df_train = df_train.drop_duplicates()
    
    for col in df_train.select_dtypes(include=[np.number]).columns:
        df_train[col].fillna(df_train[col].median(), inplace=True)
    
    le = LabelEncoder()
    df_train['user_id_encoded'] = le.fit_transform(df_train['user_id'])
    
    feature_cols = ['files_accessed', 'failed_logins', 'data_outbound_mb', 
                   'hour_of_day', 'cpu_usage', 'file_encryption_rate', 
                   'session_duration', 'user_id_encoded']
    X_train = df_train[feature_cols].values
    
    ensemble = EnsembleAnomalyDetector(contamination=0.03)
    ensemble.fit(X_train)
    
    print("2. Simulating incoming logs...")
    
    # Simulate 10 incoming logs
    for i in range(10):
        # New log arrives
        new_log = generate_banking_data(1)
        new_log = new_log.drop_duplicates()
        
        for col in new_log.select_dtypes(include=[np.number]).columns:
            new_log[col].fillna(new_log[col].median(), inplace=True)
        
        new_log['user_id_encoded'] = le.transform(new_log['user_id'])
        X_new = new_log[feature_cols].values
        
        # Predict
        pred, score, _, _ = ensemble.predict(X_new)
        
        if pred[0] == 1:
            print(f"   Log {i+1}: 🚨 ANOMALY DETECTED! (score: {score[0]:.4f})")
        else:
            print(f"   Log {i+1}: ✅ Normal (score: {score[0]:.4f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "ENSEMBLE RANSOMWARE DETECTION - USAGE EXAMPLES" + " "*12 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Run all examples
        example_basic_ensemble()
        example_weighted_ensemble()
        example_model_subset()
        example_voting_strategies()
        example_save_load()
        example_realtime_prediction()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Modify examples for your use case")
        print("  2. Integrate with your data pipeline")
        print("  3. Run 'streamlit run app.py' for interactive dashboard")
        print("  4. Check ensemble_models.py for more advanced features")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()