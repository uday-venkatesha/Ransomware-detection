#!/usr/bin/env python3
"""
Quick Pipeline Runner - Simplified version without TensorFlow dependencies
Runs only Isolation Forest model to avoid threading issues
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛡️  RANSOMWARE DETECTION - QUICK RUN")
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

# STEP 3: Train Model
print("🤖 Step 3: Training Model...")
print("-"*70)

# Prepare features
exclude_cols = ['user_id', 'user_id_original', 'timestamp']
feature_cols = [col for col in df.columns if col not in exclude_cols 
                and df[col].dtype in [np.float64, np.int64]]

X = df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Training on {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features...")

# Train Isolation Forest
print("Training Isolation Forest...")
model = IsolationForest(
    contamination=0.03,
    random_state=42,
    n_estimators=100,
    n_jobs=1,  # Single thread to avoid issues
    verbose=0
)

model.fit(X_scaled)
print("✅ Model trained successfully\n")

# STEP 4: Predict and Evaluate
print("📈 Step 4: Generating Predictions...")
print("-"*70)

# Predict
anomaly_scores = -model.score_samples(X_scaled)
predictions = model.predict(X_scaled)
is_anomaly = (predictions == -1).astype(int)

# Create ground truth
print("Creating ground truth labels for evaluation...")
true_labels = np.zeros(len(df))
conditions = (
    (df['file_encryption_rate'] > 10) |
    ((df['failed_logins'] > 8) & (df['session_duration'] < 500)) |
    ((df['data_outbound_mb'] > 400) & (df['hour_of_day'].isin([0,1,2,3,22,23]))) |
    ((df['files_accessed'] > 500) & (df['cpu_usage'] > 70))
)
true_labels[conditions] = 1

# Calculate metrics
precision = precision_score(true_labels, is_anomaly, zero_division=0)
recall = recall_score(true_labels, is_anomaly, zero_division=0)
f1 = f1_score(true_labels, is_anomaly, zero_division=0)

# Add results to dataframe
df['anomaly_score'] = anomaly_scores
df['is_anomaly'] = is_anomaly
df['true_label'] = true_labels

# Save results
df.to_csv("banking_logs_with_scores.csv", index=False)

print("\n" + "="*70)
print("📊 RESULTS SUMMARY")
print("="*70)
print(f"Model Performance:")
print(f"  • Precision:  {precision:.2%}")
print(f"  • Recall:     {recall:.2%}")
print(f"  • F1-Score:   {f1:.2%}")
print(f"  • Anomalies:  {is_anomaly.sum()} / {len(df)} ({is_anomaly.mean()*100:.1f}%)")
print(f"  • True Anomalies: {int(true_labels.sum())}")
print()

print("📁 Output Files Created:")
print("  • banking_logs_raw.csv")
print("  • banking_logs_clean.csv")
print("  • banking_logs_with_scores.csv")
print()

print("🔝 Top 5 Anomalous Sessions:")
print("-"*70)
top_5 = df.nlargest(5, 'anomaly_score')[[
    'user_id_original', 'files_accessed', 'failed_logins',
    'data_outbound_mb', 'file_encryption_rate', 'anomaly_score', 'is_anomaly'
]]
print(top_5.to_string(index=False))
print()

print("="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print("\nNext Steps:")
print("  1. Review: banking_logs_with_scores.csv")
print("  2. Run Dashboard: streamlit run app.py")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)