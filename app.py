import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import time

# Import custom modules
from generate_data import generate_banking_data
from preprocess import DataPreprocessor
from model import AnomalyDetector, create_ground_truth_labels

# Page configuration
st.set_page_config(
    page_title="🛡️ Ransomware Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Title
st.markdown("<h1 class='main-header'>🛡️ Proactive Ransomware Detection System</h1>", unsafe_allow_html=True)
st.markdown("**Unsupervised Machine Learning for Banking Security**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=100)
    st.title("⚙️ Configuration")
    
    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Generate New Dataset", "Upload CSV File"]
    )
    
    if data_source == "Generate New Dataset":
        n_records = st.slider("Number of records", 1000, 20000, 10000, 1000)
        if st.button("🚀 Generate Data", type="primary"):
            with st.spinner("Generating synthetic banking logs..."):
                df_raw = generate_banking_data(n_records)
                st.session_state.df_raw = df_raw
                st.session_state.data_generated = True
                st.success(f"✅ Generated {len(df_raw)} records!")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df_raw
            st.session_state.data_generated = True
            st.success(f"✅ Loaded {len(df_raw)} records!")
    
    st.markdown("---")
    
    st.subheader("🤖 Model Settings")
    model_type = st.selectbox(
        "Choose model:",
        ["Isolation Forest", "Autoencoder"]
    )
    
    contamination = st.slider(
        "Expected anomaly rate (%)",
        1.0, 10.0, 3.0, 0.5
    ) / 100
    
    threshold = st.slider(
        "Anomaly score threshold",
        0.0, 1.0, 0.5, 0.05
    )

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🔧 Preprocessing", 
    "🎯 Model Training", 
    "📈 Results & Analysis",
    "🔴 Real-time Simulation"
])

# Tab 1: Data Overview
with tab1:
    st.header("📊 Dataset Overview")
    
    if st.session_state.data_generated:
        df_raw = st.session_state.df_raw
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_raw):,}")
        with col2:
            st.metric("Features", len(df_raw.columns))
        with col3:
            st.metric("Missing Values", df_raw.isnull().sum().sum())
        with col4:
            st.metric("Unique Users", df_raw['user_id'].nunique())
        
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(20), use_container_width=True)
        
        st.subheader("Data Quality Issues")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values by Column**")
            missing = df_raw.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                fig = px.bar(x=missing.index, y=missing.values, 
                           labels={'x': 'Column', 'y': 'Missing Count'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found")
        
        with col2:
            st.write("**Data Statistics**")
            st.dataframe(df_raw.describe(), use_container_width=True)
        
    else:
        st.info("👈 Please generate or upload data from the sidebar")

# Tab 2: Preprocessing
with tab2:
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.data_generated:
        if st.button("▶️ Run Preprocessing Pipeline", type="primary"):
            with st.spinner("Preprocessing data..."):
                preprocessor = DataPreprocessor()
                df_scaled, df_clean = preprocessor.preprocess(st.session_state.df_raw)
                
                st.session_state.df_scaled = df_scaled
                st.session_state.df_clean = df_clean
                st.session_state.preprocessor = preprocessor
                st.session_state.data_preprocessed = True
                
                st.success("✅ Preprocessing complete!")
        
        if st.session_state.data_preprocessed:
            st.subheader("📋 Preprocessing Log")
            log_text = st.session_state.preprocessor.get_preprocessing_summary()
            st.text_area("Processing steps:", log_text, height=300)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before Preprocessing")
                st.metric("Records", len(st.session_state.df_raw))
                st.metric("Missing Values", st.session_state.df_raw.isnull().sum().sum())
            
            with col2:
                st.subheader("After Preprocessing")
                st.metric("Records", len(st.session_state.df_clean))
                st.metric("Missing Values", st.session_state.df_clean.isnull().sum().sum())
            
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)
            
            # Download button
            csv = st.session_state.df_clean.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Data",
                data=csv,
                file_name="banking_logs_clean.csv",
                mime="text/csv"
            )
    else:
        st.info("👈 Please generate or upload data first")

# Tab 3: Model Training
with tab3:
    st.header("🎯 Anomaly Detection Model")
    
    if st.session_state.data_preprocessed:
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                # Initialize detector
                model_type_key = 'isolation_forest' if model_type == 'Isolation Forest' else 'autoencoder'
                detector = AnomalyDetector(model_type=model_type_key, contamination=contamination)
                
                # Train
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                detector.train(st.session_state.df_scaled)
                
                # Create ground truth for evaluation
                true_labels = create_ground_truth_labels(st.session_state.df_clean)
                
                # Evaluate
                results = detector.evaluate(st.session_state.df_scaled, true_labels)
                
                # Save to session state
                st.session_state.detector = detector
                st.session_state.results = results
                st.session_state.true_labels = true_labels
                st.session_state.model_trained = True
                
                st.success("✅ Model training complete!")
        
        if st.session_state.model_trained:
            results = st.session_state.results
            
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{results.get('precision', 0):.2%}")
            with col2:
                st.metric("Recall", f"{results.get('recall', 0):.2%}")
            with col3:
                st.metric("F1-Score", f"{results.get('f1_score', 0):.2%}")
            with col4:
                st.metric("Anomalies Detected", f"{results['n_anomalies']}")
            
            # Confusion Matrix
            if 'confusion_matrix' in results:
                st.subheader("Confusion Matrix")
                cm = results['confusion_matrix']
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Normal', 'Anomaly'],
                              y=['Normal', 'Anomaly'],
                              text_auto=True,
                              color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Please preprocess data first")

# Tab 4: Results & Analysis
with tab4:
    st.header("📈 Detection Results & Analysis")
    
    if st.session_state.model_trained:
        results = st.session_state.results
        df_clean = st.session_state.df_clean.copy()
        
        # Add scores to dataframe
        df_clean['anomaly_score'] = results['anomaly_scores']
        df_clean['is_anomaly'] = results['predictions']
        
        # Anomaly Score Distribution
        st.subheader("📊 Anomaly Score Distribution")
        fig = px.histogram(df_clean, x='anomaly_score', 
                          color='is_anomaly',
                          nbins=50,
                          labels={'is_anomaly': 'Anomaly'},
                          title="Distribution of Anomaly Scores")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Anomalies
        st.subheader("🚨 Top 10 Anomalous Sessions")
        top_anomalies = df_clean.nlargest(10, 'anomaly_score')[[
            'user_id_original', 'files_accessed', 'failed_logins',
            'data_outbound_mb', 'cpu_usage', 'file_encryption_rate',
            'anomaly_score'
        ]]
        st.dataframe(top_anomalies, use_container_width=True)
        
        # Feature Correlation Heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['user_id_encoded', 'is_anomaly']]
        
        corr_matrix = df_clean[numerical_cols].corr()
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        st.subheader("🔍 Anomaly Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_clean, 
                           x='file_encryption_rate', 
                           y='cpu_usage',
                           color='is_anomaly',
                           title="File Encryption vs CPU Usage",
                           labels={'is_anomaly': 'Anomaly'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df_clean,
                           x='data_outbound_mb',
                           y='failed_logins',
                           color='is_anomaly',
                           title="Data Outbound vs Failed Logins",
                           labels={'is_anomaly': 'Anomaly'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        csv = df_clean.to_csv(index=False)
        st.download_button(
            label="📥 Download Results with Anomaly Scores",
            data=csv,
            file_name="banking_logs_with_anomaly_scores.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Please train model first")

# Tab 5: Real-time Simulation
with tab5:
    st.header("🔴 Real-time Log Simulation")
    st.markdown("Simulate live banking activity monitoring")
    
    if st.session_state.model_trained:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("▶️ Start Simulation", type="primary"):
                st.session_state.simulation_running = True
            
            if st.button("⏸️ Stop Simulation"):
                st.session_state.simulation_running = False
        
        if 'simulation_running' in st.session_state and st.session_state.simulation_running:
            placeholder = st.empty()
            
            for i in range(10):
                # Generate new log entry
                new_log = generate_banking_data(1)
                
                # Preprocess
                preprocessor = DataPreprocessor()
                new_scaled, new_clean = preprocessor.preprocess(new_log)
                
                # Predict
                detector = st.session_state.detector
                anomaly_scores, predictions = detector.predict(new_scaled)
                
                # Display
                with placeholder.container():
                    st.subheader(f"📡 Live Log #{i+1}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("User", new_clean['user_id_original'].values[0])
                    with col2:
                        st.metric("Anomaly Score", f"{anomaly_scores[0]:.3f}")
                    with col3:
                        if predictions[0] == 1:
                            st.error("🚨 ANOMALY DETECTED!")
                        else:
                            st.success("✅ Normal Activity")
                    
                    st.dataframe(new_clean, use_container_width=True)
                
                time.sleep(2)
            
            st.session_state.simulation_running = False
            st.success("Simulation complete!")
    else:
        st.info("👈 Please train model first")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛡️ Proactive Ransomware Detection System | "
    "Built with Streamlit & Scikit-learn | "
    "Unsupervised Machine Learning for Banking Security"
    "</div>",
    unsafe_allow_html=True
)