#!/usr/bin/env python3
"""
Quick Pipeline Runner - Enhanced with Multi-Model Ensemble
Demonstrates improved detection using ensemble methods
"""

import sys
import os

# Prevent threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from ensemble_models import EnsembleAnomalyDetector, compare_ensemble_strategies

print("="*70)
print("🛡️  RANSOMWARE DETECTION - ENSEMBLE QUICK RUN")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# STEP 1: Generate Data
print("📊 Step 1: Generating Data...")
print("-"*70)

from generate_data import generate_banking_data

df_raw = generate_banking_data(10000)
df_raw.to_csv("banking_logs_raw.csv", index=False)
print(f"✅ Generated {len(df_raw)} records\n")

# STEP 2: Preprocess
print("🔧 Step 2: Preprocessing...")
print("-"*70)

df = df_raw.copy()

# Handle missing values
print("Handling missing values...")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fix invalid ranges
print("Fixing invalid ranges...")
if 'cpu_usage' in df.columns:
    df.loc[df['cpu_usage'] > 100, 'cpu_usage'] = 100
    df.loc[df['cpu_usage'] < 0, 'cpu_usage'] = 0

if 'data_outbound_mb' in df.columns:
    df.loc[df['data_outbound_mb'] < 0, 'data_outbound_mb'] = 0

if 'hour_of_day' in df.columns:
    df.loc[df['hour_of_day'] > 23, 'hour_of_day'] = 23
    df.loc[df['hour_of_day'] < 0, 'hour_of_day'] = 0

if 'failed_logins' in df.columns:
    df.loc[df['failed_logins'] < 0, 'failed_logins'] = 0

# Remove duplicates
initial_len = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_len - len(df)} duplicates")

# Encode user_id
print("Encoding categorical variables...")
df['user_id_original'] = df['user_id']
le = LabelEncoder()
df['user_id_encoded'] = le.fit_transform(df['user_id'])

df.to_csv("banking_logs_clean.csv", index=False)
print(f"✅ Cleaned {len(df)} records\n")

# STEP 3: Train Ensemble Model
print("🤖 Step 3: Training Multi-Model Ensemble...")
print("-"*70)

# Prepare features
exclude_cols = ['user_id', 'user_id_original', 'timestamp']
feature_cols = [col for col in df.columns if col not in exclude_cols 
                and df[col].dtype in [np.float64, np.int64]]

X = df[feature_cols].values

print(f"Training on {X.shape[0]} samples with {X.shape[1]} features...")

# Create ground truth for evaluation
print("\nCreating ground truth labels...")
true_labels = np.zeros(len(df))
conditions = (
    (df['file_encryption_rate'] > 10) |
    ((df['failed_logins'] > 8) & (df['session_duration'] < 500)) |
    ((df['data_outbound_mb'] > 400) & (df['hour_of_day'].isin([0,1,2,3,22,23]))) |
    ((df['files_accessed'] > 500) & (df['cpu_usage'] > 70))
)
true_labels[conditions] = 1
print(f"True anomalies in dataset: {int(true_labels.sum())} ({true_labels.mean()*100:.1f}%)")

# Train ensemble
print("\nInitializing Ensemble Detector...")
ensemble = EnsembleAnomalyDetector(
    contamination=0.03,
    voting='soft',
    weights=None  # Equal weights initially
)

ensemble.fit(X)

# Get predictions
print("\n📈 Step 4: Generating Ensemble Predictions...")
print("-"*70)

predictions, scores, individual_preds, individual_scores = ensemble.predict(X)

# Evaluate
metrics = ensemble.evaluate(X, true_labels)

# Add results to dataframe
df['ensemble_score'] = scores
df['ensemble_prediction'] = predictions
df['true_label'] = true_labels

# Add individual model predictions
for model_name in ensemble.model_names:
    df[f'{model_name}_pred'] = individual_preds[model_name]
    df[f'{model_name}_score'] = individual_scores[model_name]

# Save results
df.to_csv("banking_logs_ensemble_results.csv", index=False)
ensemble.save("ensemble_model.pkl")

print("\n" + "="*70)
print("📊 ENSEMBLE RESULTS SUMMARY")
print("="*70)

print("\n🎯 Ensemble Performance:")
print(f"  • Precision:  {metrics['ensemble']['precision']:.2%}")
print(f"  • Recall:     {metrics['ensemble']['recall']:.2%}")
print(f"  • F1-Score:   {metrics['ensemble']['f1_score']:.2%}")
print(f"  • Anomalies:  {predictions.sum()} / {len(df)} ({predictions.mean()*100:.1f}%)")

print("\n📈 Individual Model Performance:")
print("-"*70)
for model_name, model_metrics in metrics['individual_models'].items():
    print(f"\n{model_name.upper()}:")
    print(f"  • Precision: {model_metrics['precision']:.2%}")
    print(f"  • Recall:    {model_metrics['recall']:.2%}")
    print(f"  • F1-Score:  {model_metrics['f1_score']:.2%}")

# Compare with single model (Isolation Forest only)
print("\n" + "="*70)
print("📊 ENSEMBLE vs SINGLE MODEL COMPARISON")
print("="*70)

from sklearn.ensemble import IsolationForest

# Train single IF model for comparison
print("\nTraining single Isolation Forest for comparison...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
single_model = IsolationForest(contamination=0.03, random_state=42, n_estimators=100, n_jobs=1)
single_model.fit(X_scaled)
single_preds = (single_model.predict(X_scaled) == -1).astype(int)

single_precision = precision_score(true_labels, single_preds, zero_division=0)
single_recall = recall_score(true_labels, single_preds, zero_division=0)
single_f1 = f1_score(true_labels, single_preds, zero_division=0)

print("\nSINGLE MODEL (Isolation Forest):")
print(f"  • Precision: {single_precision:.2%}")
print(f"  • Recall:    {single_recall:.2%}")
print(f"  • F1-Score:  {single_f1:.2%}")

print("\nENSEMBLE (All Models):")
print(f"  • Precision: {metrics['ensemble']['precision']:.2%}")
print(f"  • Recall:    {metrics['ensemble']['recall']:.2%}")
print(f"  • F1-Score:  {metrics['ensemble']['f1_score']:.2%}")

improvement = (metrics['ensemble']['f1_score'] - single_f1) / single_f1 * 100
print(f"\n🚀 Ensemble Improvement: {improvement:+.1f}%")

print("\n📁 Output Files Created:")
print("  • banking_logs_raw.csv")
print("  • banking_logs_clean.csv")
print("  • banking_logs_ensemble_results.csv")
print("  • ensemble_model.pkl")

print("\n🔝 Top 10 Anomalous Sessions (by Ensemble):")
print("-"*70)
top_10 = df.nlargest(10, 'ensemble_score')[[
    'user_id_original', 'files_accessed', 'failed_logins',
    'data_outbound_mb', 'file_encryption_rate', 'cpu_usage',
    'ensemble_score', 'ensemble_prediction', 'true_label'
]]
print(top_10.to_string(index=False))

print("\n" + "="*70)
print("✅ ENSEMBLE PIPELINE COMPLETE!")
print("="*70)
print("\nNext Steps:")
print("  1. Review: banking_logs_ensemble_results.csv")
print("  2. Run Enhanced Dashboard: streamlit run app.py")
print("  3. Compare model predictions in the results file")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)