import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_banking_data(n_records=10000):
    """
    Generate synthetic banking system logs with normal and anomalous activities.
    Includes noise, missing values, and data quality issues.
    """
    
    data = {
        'user_id': [],
        'files_accessed': [],
        'failed_logins': [],
        'data_outbound_mb': [],
        'hour_of_day': [],
        'cpu_usage': [],
        'file_encryption_rate': [],
        'session_duration': []
    }
    
    # Generate user IDs
    user_ids = [f"USR{str(i).zfill(4)}" for i in range(1, 201)]
    
    for i in range(n_records):
        # Determine if this is an anomaly (hidden ~3% anomalous records)
        is_anomaly = random.random() < 0.03
        
        if is_anomaly:
            # Anomalous behavior patterns
            anomaly_type = random.choice(['ransomware', 'data_exfil', 'brute_force'])
            
            if anomaly_type == 'ransomware':
                # High file encryption, high CPU, many files accessed
                data['user_id'].append(random.choice(user_ids))
                data['files_accessed'].append(np.random.randint(200, 1500))
                data['failed_logins'].append(np.random.randint(0, 5))
                data['data_outbound_mb'].append(np.random.uniform(50, 500))
                data['hour_of_day'].append(np.random.randint(0, 24))
                data['cpu_usage'].append(np.random.uniform(75, 99))
                data['file_encryption_rate'].append(np.random.uniform(15, 50))
                data['session_duration'].append(np.random.randint(3600, 14400))
                
            elif anomaly_type == 'data_exfil':
                # High data outbound, unusual hours
                data['user_id'].append(random.choice(user_ids))
                data['files_accessed'].append(np.random.randint(100, 800))
                data['failed_logins'].append(np.random.randint(0, 3))
                data['data_outbound_mb'].append(np.random.uniform(500, 2000))
                data['hour_of_day'].append(np.random.choice([0, 1, 2, 3, 4, 22, 23]))
                data['cpu_usage'].append(np.random.uniform(40, 70))
                data['file_encryption_rate'].append(np.random.uniform(0, 2))
                data['session_duration'].append(np.random.randint(1800, 7200))
                
            else:  # brute_force
                # Many failed logins, short sessions
                data['user_id'].append(random.choice(user_ids))
                data['files_accessed'].append(np.random.randint(0, 10))
                data['failed_logins'].append(np.random.randint(10, 50))
                data['data_outbound_mb'].append(np.random.uniform(0, 5))
                data['hour_of_day'].append(np.random.randint(0, 24))
                data['cpu_usage'].append(np.random.uniform(10, 30))
                data['file_encryption_rate'].append(0)
                data['session_duration'].append(np.random.randint(30, 300))
        else:
            # Normal behavior
            data['user_id'].append(random.choice(user_ids))
            data['files_accessed'].append(np.random.randint(1, 150))
            data['failed_logins'].append(np.random.choice([0, 0, 0, 0, 1, 1, 2]))
            data['data_outbound_mb'].append(np.random.uniform(0, 50))
            data['hour_of_day'].append(np.random.choice(list(range(7, 19)) * 3 + list(range(0, 24))))
            data['cpu_usage'].append(np.random.uniform(10, 60))
            data['file_encryption_rate'].append(np.random.uniform(0, 1))
            data['session_duration'].append(np.random.randint(300, 3600))
    
    df = pd.DataFrame(data)
    
    # Add noise and data quality issues
    # 1. Missing values (5% random)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    missing_cols = np.random.choice(df.columns[1:], size=len(missing_indices))
    for idx, col in zip(missing_indices, missing_cols):
        df.loc[idx, col] = np.nan
    
    # 2. Invalid values (negative, >100% CPU, etc.) - 2%
    error_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    for idx in error_indices:
        error_type = random.choice(['negative_data', 'high_cpu', 'invalid_hour'])
        if error_type == 'negative_data':
            df.loc[idx, 'data_outbound_mb'] = -np.random.uniform(1, 50)
        elif error_type == 'high_cpu':
            df.loc[idx, 'cpu_usage'] = np.random.uniform(100, 150)
        elif error_type == 'invalid_hour':
            df.loc[idx, 'hour_of_day'] = np.random.choice([24, 25, -1])
    
    # 3. Add some duplicates (1%)
    dup_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 4. Add timestamp
    start_date = datetime.now() - timedelta(days=30)
    df['timestamp'] = [start_date + timedelta(seconds=np.random.randint(0, 30*24*3600)) 
                       for _ in range(len(df))]
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic banking system logs...")
    df = generate_banking_data(10000)
    
    output_file = "banking_logs_raw.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Saved to {output_file}")
    print(f"\nDataset Preview:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nBasic Statistics:")
    print(df.describe())