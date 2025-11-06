import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Robust preprocessing pipeline for banking logs data.
    Handles missing values, invalid ranges, duplicates, scaling, and encoding.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.preprocessing_log = []
        
    def log(self, message):
        """Log preprocessing steps"""
        self.preprocessing_log.append(message)
        print(f"[PREPROCESS] {message}")
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        self.log(f"Missing values before: {df.isnull().sum().sum()}")
        
        # Strategy: Fill numerical with median, categorical with mode
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.log(f"  Filled {col} missing values with median: {median_val:.2f}")
        
        self.log(f"Missing values after: {df.isnull().sum().sum()}")
        return df
    
    def fix_invalid_ranges(self, df):
        """Fix invalid data ranges"""
        errors_fixed = 0
        
        # CPU usage should be 0-100
        if 'cpu_usage' in df.columns:
            invalid_cpu = (df['cpu_usage'] < 0) | (df['cpu_usage'] > 100)
            if invalid_cpu.sum() > 0:
                df.loc[df['cpu_usage'] > 100, 'cpu_usage'] = 100
                df.loc[df['cpu_usage'] < 0, 'cpu_usage'] = 0
                errors_fixed += invalid_cpu.sum()
                self.log(f"  Fixed {invalid_cpu.sum()} invalid CPU usage values")
        
        # Data outbound should be non-negative
        if 'data_outbound_mb' in df.columns:
            invalid_data = df['data_outbound_mb'] < 0
            if invalid_data.sum() > 0:
                df.loc[invalid_data, 'data_outbound_mb'] = 0
                errors_fixed += invalid_data.sum()
                self.log(f"  Fixed {invalid_data.sum()} negative data outbound values")
        
        # Hour of day should be 0-23
        if 'hour_of_day' in df.columns:
            invalid_hour = (df['hour_of_day'] < 0) | (df['hour_of_day'] > 23)
            if invalid_hour.sum() > 0:
                df.loc[df['hour_of_day'] > 23, 'hour_of_day'] = 23
                df.loc[df['hour_of_day'] < 0, 'hour_of_day'] = 0
                errors_fixed += invalid_hour.sum()
                self.log(f"  Fixed {invalid_hour.sum()} invalid hour values")
        
        # Failed logins should be non-negative
        if 'failed_logins' in df.columns:
            invalid_logins = df['failed_logins'] < 0
            if invalid_logins.sum() > 0:
                df.loc[invalid_logins, 'failed_logins'] = 0
                errors_fixed += invalid_logins.sum()
                self.log(f"  Fixed {invalid_logins.sum()} negative failed login values")
        
        self.log(f"Total invalid values fixed: {errors_fixed}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        initial_count = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        self.log(f"Removed {duplicates_removed} duplicate rows")
        return df_clean
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        if 'user_id' in df.columns:
            # Store original user_id for reference
            df['user_id_original'] = df['user_id']
            df['user_id_encoded'] = self.label_encoder.fit_transform(df['user_id'])
            self.log(f"Encoded user_id: {df['user_id'].nunique()} unique users")
        
        return df
    
    def scale_features(self, df):
        """Scale and normalize numerical features"""
        # Select numerical columns for scaling (exclude encoded categorical)
        exclude_cols = ['user_id', 'user_id_original', 'timestamp']
        numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                         if col not in exclude_cols]
        
        if len(numerical_cols) > 0:
            df_scaled = df.copy()
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.log(f"Scaled {len(numerical_cols)} numerical features")
            return df_scaled
        
        return df
    
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        self.log("=" * 50)
        self.log("Starting Data Preprocessing Pipeline")
        self.log("=" * 50)
        self.log(f"Initial dataset shape: {df.shape}")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Fix invalid ranges
        df = self.fix_invalid_ranges(df)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 4: Encode categorical
        df = self.encode_categorical(df)
        
        # Step 5: Scale features
        df_scaled = self.scale_features(df)
        
        self.log(f"Final dataset shape: {df_scaled.shape}")
        self.log("=" * 50)
        self.log("Preprocessing Complete!")
        self.log("=" * 50)
        
        return df_scaled, df  # Return both scaled and unscaled versions
    
    def get_preprocessing_summary(self):
        """Return preprocessing log as formatted string"""
        return "\n".join(self.preprocessing_log)

if __name__ == "__main__":
    # Load raw data
    print("Loading raw banking logs...")
    df_raw = pd.read_csv("banking_logs_raw.csv")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_scaled, df_clean = preprocessor.preprocess(df_raw)
    
    # Save cleaned data
    df_clean.to_csv("banking_logs_clean.csv", index=False)
    df_scaled.to_csv("banking_logs_scaled.csv", index=False)
    
    print("\n✓ Cleaned data saved to banking_logs_clean.csv")
    print("✓ Scaled data saved to banking_logs_scaled.csv")
    
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(preprocessor.get_preprocessing_summary())