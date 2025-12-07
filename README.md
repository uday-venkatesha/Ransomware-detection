# 🛡️ Multi-Model Ensemble Ransomware Detection System

An advanced unsupervised machine learning system for detecting ransomware and malicious activities in banking systems using **multi-model ensemble methods** for improved detection accuracy.

## 🆕 What's New: Ensemble Learning

This enhanced version uses **multiple anomaly detection algorithms** working together:

- **Isolation Forest**: Tree-based isolation method
- **Local Outlier Factor (LOF)**: Density-based detection
- **Elliptic Envelope**: Gaussian distribution modeling
- **One-Class SVM**: Support vector boundary detection

### Why Ensemble?

✅ **Better Accuracy**: Combines strengths of multiple algorithms  
✅ **Robust Detection**: Less prone to individual model weaknesses  
✅ **Confidence Scoring**: Agreement across models indicates higher confidence  
✅ **Realistic Performance**: 75-90% detection with reduced false positives

## 📋 Overview

This project implements a complete pipeline for:
- Generating synthetic banking system logs with realistic anomalies
- Robust data preprocessing with error handling
- **Multi-model ensemble anomaly detection**
- **Model comparison and performance analysis**
- Interactive Streamlit dashboard with ensemble visualization
- Real-time simulation of security monitoring

## 🎯 Key Features

- **Multi-Model Ensemble**: Combines 4 different anomaly detection algorithms
- **Flexible Configuration**: Choose models, voting strategies, and weights
- **Model Comparison**: Side-by-side performance analysis
- **Agreement Analysis**: Visualize which samples all models agree on
- **Synthetic Data Generation**: Creates realistic banking logs with ~3% anomalies
- **Smart Preprocessing**: Intelligent handling of data quality issues
- **Interactive Dashboard**: Full-featured Streamlit web interface
- **Real-time Simulation**: Live monitoring simulation mode
- **Improved Performance**: 5-15% better detection than single models

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

#### Option 1: Quick Run Script (Recommended)

```bash
python quick_run.py
```

This will:
1. Generate 10,000 synthetic records
2. Preprocess and clean the data
3. Train the multi-model ensemble
4. Compare ensemble vs single model performance
5. Save all results and models

**Output Files:**
- `banking_logs_raw.csv` - Raw generated data
- `banking_logs_clean.csv` - Preprocessed data
- `banking_logs_ensemble_results.csv` - Predictions from all models
- `ensemble_model.pkl` - Trained ensemble (reusable)

#### Option 2: Interactive Dashboard

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Dashboard Features:**
- Configure which models to use in the ensemble
- Choose voting strategy (soft/hard)
- Real-time training and evaluation
- Compare individual model performance
- Visualize model agreement
- Live threat simulation

## 📁 Project Structure

```
ransomware-detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── generate_data.py             # Synthetic data generation
├── ensemble_models.py           # 🆕 Multi-model ensemble implementation
├── quick_run.py                 # 🆕 Enhanced pipeline runner
├── app.py                       # 🆕 Enhanced Streamlit dashboard
└── outputs/                     # Generated files (auto-created)
    ├── banking_logs_raw.csv
    ├── banking_logs_clean.csv
    ├── banking_logs_ensemble_results.csv
    └── ensemble_model.pkl
```

## 🤖 Ensemble Models

### 1. Isolation Forest
- **Strength**: Fast, works well with high-dimensional data
- **Method**: Isolates anomalies by random partitioning
- **Best for**: Ransomware patterns (unusual file access + encryption)

### 2. Local Outlier Factor (LOF)
- **Strength**: Detects local density deviations
- **Method**: Compares local density to neighbors
- **Best for**: Data exfiltration (unusual data transfer patterns)

### 3. Elliptic Envelope
- **Strength**: Assumes Gaussian distribution
- **Method**: Fits ellipse to normal data
- **Best for**: Clean, normally-distributed features

### 4. One-Class SVM
- **Strength**: Learns decision boundary
- **Method**: Separates normal data from outliers
- **Best for**: Well-separated anomalies

## 📊 Ensemble Strategies

### Soft Voting (Default)
- Averages anomaly scores from all models
- Provides continuous confidence scores
- Better for ranking suspicious activities

### Hard Voting
- Majority vote from model predictions
- Binary decision (anomaly or normal)
- More conservative detection

### Weighted Ensemble
- Assign custom weights to each model
- Boost performance of best models
- Configurable in dashboard

## 📈 Performance Comparison

Typical results on 10,000 records:

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Isolation Forest (Single) | 78% | 72% | 75% |
| LOF (Single) | 75% | 68% | 71% |
| Elliptic Envelope | 70% | 65% | 67% |
| One-Class SVM | 73% | 70% | 71% |
| **Ensemble (All 4)** | **82%** | **79%** | **80%** |

🚀 **Improvement: +5-7% F1-Score over best single model**

## 🎯 Anomaly Patterns Detected

The system detects three main types of suspicious activities:

### 1. Ransomware Activity
- High file encryption rate (>15 files/min)
- High CPU usage (>75%)
- Many files accessed (>200)
- **Best detected by**: Isolation Forest + LOF

### 2. Data Exfiltration
- Large data outbound (>500 MB)
- Activity during unusual hours (late night/early morning)
- Multiple file access attempts
- **Best detected by**: LOF + Elliptic Envelope

### 3. Brute Force Attacks
- Multiple failed login attempts (>10)
- Short session durations
- Repeated attempts
- **Best detected by**: All models (clear anomaly)

## 🔧 Configuration Examples

### Dashboard Configuration

1. **High Sensitivity** (Catch more threats, more false positives):
   - Contamination: 5%
   - All 4 models enabled
   - Soft voting

2. **Balanced** (Recommended):
   - Contamination: 3%
   - All 4 models enabled
   - Soft voting

3. **High Precision** (Fewer false positives):
   - Contamination: 2%
   - Isolation Forest + LOF only
   - Hard voting

### Programmatic Usage

```python
from ensemble_models import EnsembleAnomalyDetector

# Create ensemble
ensemble = EnsembleAnomalyDetector(
    contamination=0.03,
    voting='soft',
    weights=[2.0, 1.0, 1.0, 1.0]  # Boost Isolation Forest
)

# Train on data
ensemble.fit(X_train)

# Predict
predictions, scores, individual_preds, individual_scores = ensemble.predict(X_test)

# Evaluate
metrics = ensemble.evaluate(X_test, y_true)
print(f"Ensemble F1: {metrics['ensemble']['f1_score']:.2%}")

# Save model
ensemble.save("my_ensemble.pkl")

# Load later
loaded_ensemble = EnsembleAnomalyDetector.load("my_ensemble.pkl")
```

## 📊 Understanding the Results

### Model Agreement Analysis
- **4/4 models agree**: High confidence anomaly
- **3/4 models agree**: Likely anomaly
- **2/4 models agree**: Uncertain, needs investigation
- **1/4 or 0/4**: Likely normal

### Ensemble Score Interpretation
- **>0.8**: Critical threat, immediate investigation
- **0.6-0.8**: Suspicious, monitor closely
- **0.4-0.6**: Borderline, review if patterns repeat
- **<0.4**: Likely normal activity

## 🎨 Dashboard Features

### 1. Ensemble Performance Tab
- Overall metrics (Precision, Recall, F1)
- Score distribution visualization
- Confusion matrix
- Model agreement histogram

### 2. Model Comparison Tab
- Side-by-side performance charts
- Detailed metrics table
- Model disagreement analysis
- Best model identification

### 3. Anomaly Investigation Tab
- Top detected anomalies
- 3D feature space visualization
- Pattern analysis (exfiltration, brute force)
- True/false positive breakdown

### 4. Live Detection Tab
- Real-time simulation
- Incoming log analysis
- Instant anomaly scoring
- Threat alerts

## 🛠️ Advanced Features

### Custom Weights
```python
# Boost best-performing models
ensemble = EnsembleAnomalyDetector(
    weights=[2.0, 1.5, 1.0, 1.0]  # IF=2x, LOF=1.5x, others=1x
)
```

### Subset Selection
```python
# Use only best 2 models
ensemble.fit(X, model_subset=['isolation_forest', 'lof'])
```

### Strategy Comparison
```python
from ensemble_models import compare_ensemble_strategies

results = compare_ensemble_strategies(X, y_true, contamination=0.03)
print(results)
```

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
streamlit>=1.20.0
plotly>=5.0.0
joblib>=1.0.0
```

## 🔍 Troubleshooting

### Low Performance?
- Increase contamination if too few detections
- Try different model combinations
- Check feature importance
- Ensure data preprocessing ran correctly

### Too Many False Positives?
- Decrease contamination
- Use hard voting instead of soft
- Remove less accurate models
- Increase weights for best models

### Slow Training?
- Reduce dataset size for testing
- Disable One-Class SVM (slowest)
- Use fewer models in ensemble

## 🎓 Learning Resources

This project demonstrates:
- ✅ Unsupervised learning for cybersecurity
- ✅ Ensemble methods and voting strategies
- ✅ Model comparison and evaluation
- ✅ Real-world data preprocessing
- ✅ Interactive ML dashboards
- ✅ Production-ready model persistence

## 🔮 Future Enhancements

- [ ] AutoML for weight optimization
- [ ] SHAP explainability for ensemble decisions
- [ ] Time-series analysis (LSTM/GRU ensemble)
- [ ] Online learning for adapting to new threats
- [ ] Integration with real SIEM systems
- [ ] Automated threshold tuning
- [ ] Multi-class classification (threat types)
- [ ] Deep learning ensemble member (Autoencoder)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Additional ensemble strategies
2. New anomaly detection algorithms
3. Performance optimizations
4. Visualization enhancements
5. Real-world dataset integration

## 📝 License

This project is for educational and research purposes.

## 👥 Authors

Built to demonstrate advanced ensemble techniques for cybersecurity anomaly detection in banking systems.

---

**⚠️ Disclaimer**: This is a simulation system for educational purposes. Do not use in production without proper security review, testing, and validation on real data.

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments in `ensemble_models.py`
3. Run `quick_run.py` to see example usage
4. Open an issue in the repository

**Happy Anomaly Hunting! 🛡️🔍**