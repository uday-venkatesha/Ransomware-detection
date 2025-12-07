import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

from generate_data import generate_banking_data
from ensemble_models import EnsembleAnomalyDetector, compare_ensemble_strategies

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🛡️ Ensemble Ransomware Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #E63946;
        text-align: center;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #457B9D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# --- HELPER FUNCTIONS ---

def preprocess_data(df_raw):
    """Clean and preprocess data"""
    df = df_raw.copy()
    
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fix invalid ranges
    if 'cpu_usage' in df.columns:
        df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
    if 'data_outbound_mb' in df.columns:
        df['data_outbound_mb'] = df['data_outbound_mb'].clip(lower=0)
    if 'hour_of_day' in df.columns:
        df['hour_of_day'] = df['hour_of_day'].clip(0, 23)
    if 'failed_logins' in df.columns:
        df['failed_logins'] = df['failed_logins'].clip(lower=0)
    
    df = df.drop_duplicates()
    
    # Encode user_id
    if 'user_id' in df.columns:
        le = LabelEncoder()
        df['user_id_encoded'] = le.fit_transform(df['user_id'])
    
    return df

def generate_ground_truth(df):
    """Create ground truth labels"""
    true_labels = np.zeros(len(df))
    conditions = (
        (df['file_encryption_rate'] > 10) |
        ((df['failed_logins'] > 8) & (df['session_duration'] < 500)) |
        ((df['data_outbound_mb'] > 400) & (df['hour_of_day'].isin([0,1,2,3,22,23]))) |
        ((df['files_accessed'] > 500) & (df['cpu_usage'] > 70))
    )
    true_labels[conditions] = 1
    return true_labels

# --- UI LAYOUT ---

st.markdown("<h1 class='main-header'>🛡️ Multi-Model Ensemble Ransomware Detection</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Advanced Anomaly Detection with Ensemble Learning</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📊 Data Generation")
    n_records = st.slider("Number of Records", 1000, 20000, 5000, step=1000)
    
    st.subheader("🤖 Ensemble Settings")
    contamination = st.slider("Expected Contamination (%)", 1, 10, 3) / 100.0
    
    voting_method = st.radio(
        "Voting Strategy",
        ["soft", "hard"],
        help="Soft: Average scores | Hard: Majority vote"
    )
    
    st.subheader("🎯 Model Selection")
    use_isolation_forest = st.checkbox("Isolation Forest", value=True)
    use_lof = st.checkbox("Local Outlier Factor", value=True)
    use_elliptic = st.checkbox("Elliptic Envelope", value=True)
    use_svm = st.checkbox("One-Class SVM", value=True)
    
    model_subset = []
    if use_isolation_forest:
        model_subset.append('isolation_forest')
    if use_lof:
        model_subset.append('lof')
    if use_elliptic:
        model_subset.append('elliptic_envelope')
    if use_svm:
        model_subset.append('one_class_svm')
    
    st.divider()
    
    if st.button("🚀 Generate & Train Ensemble", type="primary", disabled=len(model_subset)==0):
        with st.spinner("Generating synthetic data..."):
            raw_data = generate_banking_data(n_records)
            st.session_state.data = raw_data
        
        with st.spinner("Preprocessing & Cleaning..."):
            clean_df = preprocess_data(raw_data)
            st.session_state.clean_data = clean_df
        
        # Prepare features
        exclude_cols = ['user_id', 'timestamp', 'user_id_encoded']
        feature_cols = [col for col in clean_df.columns if col not in exclude_cols 
                       and clean_df[col].dtype in [np.float64, np.int64]]
        X = clean_df[feature_cols].values
        
        with st.spinner(f"Training Ensemble ({len(model_subset)} models)..."):
            # Train ensemble
            ensemble = EnsembleAnomalyDetector(
                contamination=contamination,
                voting=voting_method,
                weights=None
            )
            ensemble.fit(X, model_subset=model_subset)
            st.session_state.ensemble_model = ensemble
            
            # Get predictions
            preds, scores, ind_preds, ind_scores = ensemble.predict(X)
            
            # Generate ground truth
            true_labels = generate_ground_truth(clean_df)
            
            # Evaluate
            metrics = ensemble.evaluate(X, true_labels)
            st.session_state.metrics = metrics
            
            # Store results
            clean_df['ensemble_prediction'] = preds
            clean_df['ensemble_score'] = scores
            clean_df['true_label'] = true_labels
            
            for model_name in ensemble.model_names:
                clean_df[f'{model_name}_pred'] = ind_preds[model_name]
                clean_df[f'{model_name}_score'] = ind_scores[model_name]
            
            st.session_state.predictions = clean_df
        
        st.success("✅ Ensemble Training Complete!")
    
    if len(model_subset) == 0:
        st.warning("⚠️ Select at least one model")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ensemble Performance", 
    "🔍 Model Comparison", 
    "🎯 Anomaly Investigation",
    "🔴 Live Detection"
])

# --- TAB 1: ENSEMBLE PERFORMANCE ---
with tab1:
    if st.session_state.predictions is None:
        st.info("👈 Configure and train the ensemble in the sidebar to begin")
    else:
        df = st.session_state.predictions
        metrics = st.session_state.metrics
        
        # Metrics Row
        st.subheader("🎯 Ensemble Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        n_anomalies = df['ensemble_prediction'].sum()
        ensemble_metrics = metrics['ensemble']
        
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Anomalies", f"{n_anomalies} ({n_anomalies/len(df):.1%})")
        col3.metric("Precision", f"{ensemble_metrics['precision']:.1%}")
        col4.metric("Recall", f"{ensemble_metrics['recall']:.1%}")
        col5.metric("F1-Score", f"{ensemble_metrics['f1_score']:.1%}")
        
        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Ensemble Score Distribution")
            fig_hist = px.histogram(
                df, x="ensemble_score", 
                color="ensemble_prediction",
                nbins=50,
                title="Ensemble Anomaly Scores",
                color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                labels={"ensemble_prediction": "Prediction"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_chart2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(df['true_label'], df['ensemble_prediction'])
            fig_cm = px.imshow(
                cm, 
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['Normal', 'Anomaly'],
                y=['Normal', 'Anomaly'],
                color_continuous_scale='RdYlGn_r',
                title="Ensemble Predictions vs Ground Truth"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        st.divider()
        
        # Agreement Analysis
        st.subheader("🤝 Model Agreement Analysis")
        
        # Calculate agreement
        model_cols = [col for col in df.columns if col.endswith('_pred')]
        if len(model_cols) > 1:
            agreement = df[model_cols].sum(axis=1)
            df['model_agreement'] = agreement
            
            fig_agreement = px.histogram(
                df, x="model_agreement",
                title=f"Number of Models Detecting Each Sample as Anomaly (out of {len(model_cols)})",
                labels={"model_agreement": "Number of Models Agreeing"},
                color_discrete_sequence=['#9b59b6']
            )
            st.plotly_chart(fig_agreement, use_container_width=True)
            
            # High agreement anomalies
            high_agreement = df[df['model_agreement'] >= len(model_cols) * 0.75]
            st.info(f"🎯 **{len(high_agreement)} samples** detected by ≥75% of models (high confidence)")

# --- TAB 2: MODEL COMPARISON ---
with tab2:
    if st.session_state.metrics is None:
        st.info("Please train the ensemble first")
    else:
        metrics = st.session_state.metrics
        df = st.session_state.predictions
        
        st.subheader("📈 Individual Model Performance")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_metrics in metrics['individual_models'].items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Precision': model_metrics['precision'],
                'Recall': model_metrics['recall'],
                'F1-Score': model_metrics['f1_score']
            })
        
        # Add ensemble
        comparison_data.append({
            'Model': 'ENSEMBLE',
            'Precision': metrics['ensemble']['precision'],
            'Recall': metrics['ensemble']['recall'],
            'F1-Score': metrics['ensemble']['f1_score']
        })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Visualization
        fig_comparison = go.Figure()
        
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, metric in enumerate(metrics_to_plot):
            fig_comparison.add_trace(go.Bar(
                name=metric,
                x=comp_df['Model'],
                y=comp_df[metric],
                marker_color=colors[i],
                text=[f"{val:.1%}" for val in comp_df[metric]],
                textposition='outside'
            ))
        
        fig_comparison.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1.1])
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.divider()
        
        # Detailed metrics table
        st.subheader("📊 Detailed Metrics Table")
        st.dataframe(
            comp_df.style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}'
            }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Model disagreement analysis
        st.divider()
        st.subheader("🔍 Model Disagreement Cases")
        
        model_pred_cols = [col for col in df.columns if col.endswith('_pred')]
        if len(model_pred_cols) > 1:
            # Find cases where models disagree
            df['prediction_variance'] = df[model_pred_cols].var(axis=1)
            disagreement_cases = df[df['prediction_variance'] > 0].nlargest(10, 'ensemble_score')
            
            st.write("Top 10 cases with highest model disagreement:")
            display_cols = ['user_id', 'ensemble_score', 'ensemble_prediction'] + model_pred_cols + ['true_label']
            st.dataframe(disagreement_cases[display_cols], use_container_width=True)

# --- TAB 3: ANOMALY INVESTIGATION ---
with tab3:
    if st.session_state.predictions is None:
        st.info("Please train the ensemble first")
    else:
        df = st.session_state.predictions
        anomalies = df[df['ensemble_prediction'] == 1]
        
        st.subheader("🚨 Detected Anomalies")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("Total Anomalies", len(anomalies))
        col_metric2.metric("True Positives", len(anomalies[anomalies['true_label']==1]))
        col_metric3.metric("False Positives", len(anomalies[anomalies['true_label']==0]))
        
        st.divider()
        
        # Top anomalies
        st.subheader("🔝 Top 10 Most Anomalous Sessions")
        top_anomalies = df.nlargest(10, 'ensemble_score')
        
        display_cols = ['user_id', 'files_accessed', 'failed_logins', 'data_outbound_mb', 
                       'cpu_usage', 'file_encryption_rate', 'ensemble_score', 
                       'ensemble_prediction', 'true_label']
        
        st.dataframe(
            top_anomalies[display_cols].style.background_gradient(
                subset=['ensemble_score'], cmap='Reds'
            ),
            use_container_width=True
        )
        
        st.divider()
        
        # 3D Visualization
        st.subheader("🎨 3D Feature Space Visualization")
        
        sample_size = min(2000, len(df))
        df_sample = df.sample(sample_size)
        
        fig_3d = px.scatter_3d(
            df_sample,
            x='files_accessed',
            y='file_encryption_rate',
            z='cpu_usage',
            color='ensemble_prediction',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            size='ensemble_score',
            size_max=10,
            title="Anomaly Detection in 3D Feature Space",
            labels={'ensemble_prediction': 'Prediction'},
            opacity=0.7
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Feature correlations
        col_scat1, col_scat2 = st.columns(2)
        
        with col_scat1:
            fig_s1 = px.scatter(
                df_sample, x="files_accessed", y="data_outbound_mb",
                color="ensemble_prediction",
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                title="Data Exfiltration Pattern"
            )
            st.plotly_chart(fig_s1, use_container_width=True)
        
        with col_scat2:
            fig_s2 = px.scatter(
                df_sample, x="failed_logins", y="session_duration",
                color="ensemble_prediction",
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                title="Brute Force Pattern"
            )
            st.plotly_chart(fig_s2, use_container_width=True)

# --- TAB 4: LIVE DETECTION ---
with tab4:
    st.subheader("🔴 Real-Time Threat Monitor")
    st.write("Simulate live traffic and detect anomalies in real-time using the ensemble.")
    
    col_btn, col_status = st.columns([1, 3])
    
    with col_btn:
        run_simulation = st.toggle("▶️ Start Monitoring")
    
    if run_simulation and st.session_state.ensemble_model is not None:
        placeholder = st.empty()
        
        # Get feature columns
        exclude_cols = ['user_id', 'timestamp', 'user_id_encoded', 'ensemble_prediction',
                       'ensemble_score', 'true_label'] + [col for col in st.session_state.predictions.columns 
                                                           if '_pred' in col or '_score' in col]
        feature_cols = [col for col in st.session_state.clean_data.columns 
                       if col not in exclude_cols and st.session_state.clean_data[col].dtype in [np.float64, np.int64]]
        
        for i in range(50):
            if not run_simulation:
                break
            
            # Generate new sample
            new_sample = generate_banking_data(1)
            processed = preprocess_data(new_sample)
            
            # Predict
            X_new = processed[feature_cols].values
            preds, scores, _, _ = st.session_state.ensemble_model.predict(X_new)
            
            is_threat = preds[0] == 1
            score = scores[0]
            
            with placeholder.container():
                current_time = time.strftime('%H:%M:%S')
                
                if is_threat:
                    st.error(f"⚠️ **THREAT DETECTED** at {current_time}")
                    st.metric("🚨 Anomaly Score", f"{score:.4f}", delta="HIGH RISK", delta_color="inverse")
                else:
                    st.success(f"✅ **Normal Activity** at {current_time}")
                    st.metric("✓ Anomaly Score", f"{score:.4f}", delta="Low Risk")
                
                st.write("**Incoming Log Details:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Files Accessed", int(new_sample['files_accessed'].iloc[0]))
                col2.metric("CPU Usage", f"{new_sample['cpu_usage'].iloc[0]:.1f}%")
                col3.metric("Data Out (MB)", f"{new_sample['data_outbound_mb'].iloc[0]:.1f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Encryption Rate", f"{new_sample['file_encryption_rate'].iloc[0]:.1f}/min")
                col5.metric("Failed Logins", int(new_sample['failed_logins'].iloc[0]))
                col6.metric("Session (sec)", int(new_sample['session_duration'].iloc[0]))
            
            time.sleep(1.5)
    
    elif run_simulation and st.session_state.ensemble_model is None:
        st.warning("⚠️ Please train the ensemble model first!")
    
    if not run_simulation:
        st.info("Click the toggle above to start real-time monitoring simulation")