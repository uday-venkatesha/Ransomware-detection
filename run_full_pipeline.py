#!/usr/bin/env python3
"""
Complete Pipeline Runner
Executes the full ransomware detection pipeline from data generation to model evaluation.
"""

import sys
import os
from datetime import datetime

# Ensure env vars are set early to avoid OpenMP / BLAS thread deadlocks
# (uses the previously unused "os" import so linters won't complain)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Prefer spawn start method to avoid inheriting threads across a fork (can cause libgomp mutex logs)
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set; ignore
    pass

print("="*70)
print("🛡️  PROACTIVE RANSOMWARE DETECTION SYSTEM")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Step 1: Generate Data
print("📊 STEP 1: Generating Synthetic Banking Logs")
print("-"*70)
try:
    from generate_data import generate_banking_data
    df_raw = generate_banking_data(10000)
    df_raw.to_csv("banking_logs_raw.csv", index=False)
    print(f"✅ Generated {len(df_raw)} records")
    print(f"✅ Saved to: banking_logs_raw.csv")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print()

# Step 2: Preprocess Data
print("🔧 STEP 2: Preprocessing Data")
print("-"*70)
try:
    from preprocess import DataPreprocessor
    import pandas as pd
    
    df_raw = pd.read_csv("banking_logs_raw.csv")
    preprocessor = DataPreprocessor()
    df_scaled, df_clean = preprocessor.preprocess(df_raw)
    
    df_clean.to_csv("banking_logs_clean.csv", index=False)
    df_scaled.to_csv("banking_logs_scaled.csv", index=False)
    
    print(f"✅ Cleaned {len(df_clean)} records")
    print(f"✅ Saved to: banking_logs_clean.csv and banking_logs_scaled.csv")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print()

# Step 3: Train Model
print("🤖 STEP 3: Training Anomaly Detection Model")
print("-"*70)
try:
    from model import AnomalyDetector, create_ground_truth_labels
    import pandas as pd

    # joblib parallel backend can be controlled to avoid fork/thread issues
    try:
        from joblib import parallel_backend
    except Exception:
        parallel_backend = None
    
    df_scaled = pd.read_csv("banking_logs_scaled.csv")
    df_clean = pd.read_csv("banking_logs_clean.csv")
    
    # Create ground truth for evaluation
    true_labels = create_ground_truth_labels(df_clean)
    print(f"Ground truth: {true_labels.sum()} anomalies ({true_labels.mean()*100:.1f}%)")
    
    # Train Isolation Forest (force single-threaded execution to avoid mutex/blocking issues)
    detector = AnomalyDetector(model_type='isolation_forest', contamination=0.03)
    
    # Use threading backend with n_jobs=1 for both training and evaluation to prevent libgomp mutex warnings
    if parallel_backend is not None:
        with parallel_backend('threading', n_jobs=1):
            detector.train(df_scaled)
            results = detector.evaluate(df_scaled, true_labels)
    else:
        detector.train(df_scaled)
        results = detector.evaluate(df_scaled, true_labels)
    
    # Save results
    df_clean['anomaly_score'] = results['anomaly_scores']
    df_clean['is_anomaly'] = results['predictions']
    df_clean.to_csv("banking_logs_with_scores.csv", index=False)
    
    detector.save_model("isolation_forest_model.pkl")
    
    print(f"✅ Model trained successfully")
    print(f"✅ Detected {results['n_anomalies']} anomalies")
    print(f"✅ Saved to: banking_logs_with_scores.csv")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print()

# Step 4: Display Summary
print("📈 STEP 4: Results Summary")
print("-"*70)
print(f"Model Performance:")
print(f"  • Precision:  {results.get('precision', 0):.2%}")
print(f"  • Recall:     {results.get('recall', 0):.2%}")
print(f"  • F1-Score:   {results.get('f1_score', 0):.2%}")
print(f"  • Anomalies:  {results['n_anomalies']} / {len(df_clean)} ({results['anomaly_rate']*100:.1f}%)")
print()

print("📁 Output Files:")
print("  • banking_logs_raw.csv           - Raw generated data")
print("  • banking_logs_clean.csv         - Cleaned data")
print("  • banking_logs_scaled.csv        - Scaled features for ML")
print("  • banking_logs_with_scores.csv   - Results with anomaly scores")
print("  • isolation_forest_model.pkl     - Trained model")
print()

print("🔝 Top 5 Anomalous Sessions:")
print("-"*70)
top_5 = df_clean.nlargest(5, 'anomaly_score')[[
    'user_id_original', 'files_accessed', 'failed_logins',
    'data_outbound_mb', 'file_encryption_rate', 'anomaly_score'
]]
print(top_5.to_string(index=False))
print()

print("="*70)
print("✅ Pipeline Complete!")
print("="*70)
print()
print("Next steps:")
print("  1. Review the output files in the current directory")
print("  2. Run the Streamlit dashboard: streamlit run app.py")
print("  3. Explore visualizations and real-time simulation")
print()
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)