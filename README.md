# 🛡️ Proactive Ransomware Detection System

An unsupervised machine learning system for detecting ransomware and malicious activities in banking systems using anomaly detection techniques.

## 📋 Overview

This project implements a complete pipeline for:
- Generating synthetic banking system logs with realistic anomalies
- Robust data preprocessing with error handling
- Unsupervised anomaly detection using Isolation Forest and Autoencoder
- Interactive Streamlit dashboard for visualization and analysis
- Real-time simulation of security monitoring

## 🎯 Key Features

- **Synthetic Data Generation**: Creates realistic banking logs with ~3% anomalous activity
- **Data Quality Issues**: Simulates real-world problems (missing values, invalid ranges, duplicates)
- **Smart Preprocessing**: Intelligent handling of data quality issues
- **Multiple Models**: Isolation Forest (primary) and Autoencoder (optional)
- **Interactive Dashboard**: Full-featured Streamlit web interface
- **Real-time Simulation**: Live monitoring simulation mode
- **Realistic Performance**: 70-85% detection accuracy (not artificially perfect)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ransomware-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Command Line (Step by Step)

1. **Generate Data**:
```bash
python generate_data.py
```
Output: `banking_logs_raw.csv`

2. **Preprocess Data**:
```bash
python preprocess.py
```
Output: `banking_logs_clean.csv`, `banking_logs_scaled.csv`

3. **Train Model**:
```bash
python model.py
```
Output: `banking_logs_with_scores.csv`, `isolation_forest_model.pkl`

#### Option 2: Streamlit Dashboard (All-in-One)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
ransomware-detection/
├── generate_data.py      # Synthetic data generation
├── preprocess.py         # Data preprocessing pipeline
├── model.py              # Anomaly detection models
├── app.py                # Streamlit dashboard
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── outputs/              # Generated files (created automatically)
    ├── banking_logs_raw.csv
    ├── banking_logs_clean.csv
    ├── banking_logs_scaled.csv
    ├── banking_logs_with_scores.csv
    └── isolation_forest_model.pkl
```

## 📊 Data Schema

The generated dataset includes the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `user_id` | Unique user/system identifier | String |
| `files_accessed` | Number of files accessed in session | Integer |
| `failed_logins` | Number of failed login attempts | Integer |
| `data_outbound_mb` | Amount of data transferred out (MB) | Float |
| `hour_of_day` | Time of activity (0-23) | Integer |
| `cpu_usage` | Average CPU usage (%) | Float |
| `file_encryption_rate` | Files encrypted per minute | Float |
| `session_duration` | Active session time (seconds) | Integer |
| `timestamp` | Log timestamp | DateTime |

## 🎯 Anomaly Patterns Detected

The system detects three main types of suspicious activities:

1. **Ransomware Activity**
   - High file encryption rate (>15 files/min)
   - High CPU usage (>75%)
   - Many files accessed (>200)

2. **Data Exfiltration**
   - Large data outbound (>500 MB)
   - Activity during unusual hours (late night/early morning)
   - Multiple file access attempts

3. **Brute Force Attacks**
   - Multiple failed login attempts (>10)
   - Short session durations
   - Repeated attempts

## 🤖 Model Details

### Isolation Forest (Primary Model)

- **Algorithm**: Isolation Forest from scikit-learn
- **Contamination**: 3% (configurable)
- **Parameters**:
  - n_estimators: 100
  - max_samples: auto
  - random_state: 42
- **Performance**: ~80-85% detection rate

### Autoencoder (Optional)

- **Architecture**: Encoder-Decoder neural network
- **Framework**: TensorFlow/Keras
- **Training**: 50 epochs, MSE loss
- **Detection**: Based on reconstruction error

## 📈 Using the Streamlit Dashboard

### 1. Data Overview Tab
- View raw data statistics
- Identify data quality issues
- Explore missing values and distributions

### 2. Preprocessing Tab
- Run preprocessing pipeline
- View cleaning logs
- Compare before/after statistics
- Download cleaned data

### 3. Model Training Tab
- Select model type (Isolation Forest or Autoencoder)
- Configure contamination rate
- Train model and view performance metrics
- See confusion matrix

### 4. Results & Analysis Tab
- View anomaly score distribution
- Examine top anomalous sessions
- Explore feature correlations
- Visualize anomalies in scatter plots
- Download results with scores

### 5. Real-time Simulation Tab
- Start live monitoring simulation
- See new logs analyzed in real-time
- Detect anomalies as they occur

## 🔧 Configuration

### Model Settings (in Streamlit sidebar)

- **Model Type**: Choose between Isolation Forest or Autoencoder
- **Expected Anomaly Rate**: Set contamination parameter (1-10%)
- **Anomaly Threshold**: Adjust sensitivity (0.0-1.0)

### Data Generation Settings

- **Number of Records**: 1,000 to 20,000
- **Anomaly Rate**: ~3% (hardcoded in generator, realistic)

## 📊 Performance Expectations

The system is designed for **realistic performance**, not perfect accuracy:

- **Detection Rate**: 70-85%
- **Precision**: 75-90%
- **Recall**: 65-85%
- **F1-Score**: 70-85%

This reflects real-world challenges:
- Noisy data
- Evolving attack patterns
- Legitimate edge cases that look suspicious
- Zero-day threats not seen in training

## 🛠️ Preprocessing Pipeline

The system handles various data quality issues:

1. **Missing Values**: Imputed with median (numerical)
2. **Invalid Ranges**: 
   - CPU usage capped at 100%
   - Negative values corrected to 0
   - Hour of day normalized to 0-23
3. **Duplicates**: Automatically removed
4. **Scaling**: StandardScaler normalization
5. **Encoding**: Label encoding for categorical variables

## 📦 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models
- `streamlit` - Web interface
- `plotly` - Interactive visualizations
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations
- `tensorflow` (optional) - For Autoencoder model
- `joblib` - Model persistence

## 🔍 Evaluation Metrics

Even though this is unsupervised learning, we evaluate using:

1. **Ground Truth Labels**: Created from extreme patterns (~1-2% of data)
2. **Precision**: Accuracy of anomaly predictions
3. **Recall**: Coverage of actual anomalies
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: True/false positives and negatives

## 🎨 Visualization Features

- Anomaly score distribution histograms
- Top anomalous sessions table
- Feature correlation heatmaps
- Scatter plots (encryption vs CPU, data vs logins)
- Real-time activity monitoring
- Confusion matrix heatmap

## 🚨 Anomaly Detection Logic

Anomalies are identified based on:

1. **Isolation Forest**: Records that require fewer splits to isolate
2. **Autoencoder**: High reconstruction error
3. **Threshold**: Top N% based on contamination parameter

## 💡 Tips for Best Results

1. **Data Generation**: Use 10,000+ records for better model training
2. **Contamination**: Set to realistic levels (2-5%)
3. **Threshold Tuning**: Adjust based on false positive tolerance
4. **Feature Engineering**: Key features are file_encryption_rate, cpu_usage, and data_outbound_mb
5. **Preprocessing**: Always run preprocessing before training

## 🔮 Future Enhancements

- SHAP explainability integration
- Datadog API for real alerts
- Time-series analysis
- User behavior profiling
- Multi-model ensemble
- API endpoint for production deployment

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and research purposes.

## 👥 Authors

Built as a demonstration of unsupervised machine learning for cybersecurity applications in banking systems.

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**⚠️ Disclaimer**: This is a simulation system for educational purposes. Do not use in production without proper security review and testing.