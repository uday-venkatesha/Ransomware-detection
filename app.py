import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Import the data generator from your provided file
from generate_data import generate_banking_data

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🛡️ Ransomware Detection System",
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
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# --- HELPER FUNCTIONS (Logic moved from quick_run.py) ---

def preprocess_data(df_raw):
    """
    Cleans the raw dataframe based on logic from quick_run.py
    """
    df = df_raw.copy()
    
    # 1. Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Fix invalid ranges
    if 'cpu_usage' in df.columns:
        df['cpu_usage'] = df['cpu_usage'].clip(0, 100)
    if 'data_outbound_mb' in df.columns:
        df['data_outbound_mb'] = df['data_outbound_mb'].clip(lower=0)
    if 'hour_of_day' in df.columns:
        df['hour_of_day'] = df['hour_of_day'].clip(0, 23)
    if 'failed_logins' in df.columns:
        df['failed_logins'] = df['failed_logins'].clip(lower=0)

    # 3. Remove duplicates
    df = df.drop_duplicates()
    
    # 4. Encode user_id
    if 'user_id' in df.columns:
        le = LabelEncoder()
        df['user_id_encoded'] = le.fit_transform(df['user_id'])
        
    return df

def train_and_predict(df, contamination=0.03):
    """
    Trains Isolation Forest and generates predictions
    """
    # Select features
    exclude_cols = ['user_id', 'timestamp', 'user_id_encoded', 'is_anomaly', 'anomaly_type'] 
    feature_cols = [col for col in df.columns if col not in exclude_cols 
                    and df[col].dtype in [np.float64, np.int64]]
    
    X = df[feature_cols].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Predict
    # -1 is anomaly, 1 is normal in sklearn
    # We convert to: 1 = anomaly, 0 = normal
    raw_predictions = model.predict(X_scaled)
    anomaly_scores = -model.score_samples(X_scaled)
    is_anomaly = (raw_predictions == -1).astype(int)
    
    return model, scaler, is_anomaly, anomaly_scores, feature_cols

def generate_ground_truth(df):
    """
    Recreates the ground truth logic from quick_run.py for evaluation
    """
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

st.markdown("<h1 class='main-header'>🛡️ Proactive Ransomware Detection</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Unsupervised Anomaly Detection for Banking Logs</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Data Generation")
    n_records = st.slider("Number of Records", 1000, 20000, 5000, step=1000)
    
    st.subheader("Model Settings")
    contamination = st.slider("Expected Contamination (%)", 1, 10, 3) / 100.0
    
    if st.button("🔄 Generate & Train", type="primary"):
        with st.spinner("Generating synthetic data..."):
            raw_data = generate_banking_data(n_records)
            st.session_state.data = raw_data
        
        with st.spinner("Preprocessing & Cleaning..."):
            clean_df = preprocess_data(raw_data)
            st.session_state.clean_data = clean_df
            
        with st.spinner("Training Isolation Forest..."):
            model, scaler, preds, scores, features = train_and_predict(clean_df, contamination)
            st.session_state.model = model
            st.session_state.scaler = scaler
            
            # Combine results
            clean_df['is_anomaly_pred'] = preds
            clean_df['anomaly_score'] = scores
            clean_df['true_label'] = generate_ground_truth(clean_df)
            st.session_state.predictions = clean_df
            st.session_state.features = features
            
        st.success("Pipeline Completed Successfully!")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data & Performance", "🔍 Anomaly Investigation", "🔴 Live Simulation"])

# --- TAB 1: DATA & PERFORMANCE ---
with tab1:
    if st.session_state.predictions is None:
        st.info("👈 Click 'Generate & Train' in the sidebar to start the system.")
    else:
        df = st.session_state.predictions
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        n_anomalies = df['is_anomaly_pred'].sum()
        precision = precision_score(df['true_label'], df['is_anomaly_pred'])
        recall = recall_score(df['true_label'], df['is_anomaly_pred'])
        f1 = f1_score(df['true_label'], df['is_anomaly_pred'])
        
        col1.metric("Total Records", len(df))
        col2.metric("Anomalies Detected", f"{n_anomalies} ({n_anomalies/len(df):.1%})")
        col3.metric("Precision", f"{precision:.2%}")
        col4.metric("Recall (Sensitivity)", f"{recall:.2%}")
        
        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Anomaly Score Distribution")
            fig_hist = px.histogram(df, x="anomaly_score", color="is_anomaly_pred", 
                                    nbins=50, title="Distribution of Isolation Forest Scores",
                                    color_discrete_map={0: "blue", 1: "red"},
                                    labels={"is_anomaly_pred": "Detected Anomaly"})
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_chart2:
            st.subheader("Confusion Matrix")
            # Simple confusion matrix viz
            cm_data = pd.crosstab(df['true_label'], df['is_anomaly_pred'], 
                                  rownames=['Actual'], colnames=['Predicted'])
            fig_cm = px.imshow(cm_data, text_auto=True, color_continuous_scale='Blues',
                               title="Confusion Matrix (Actual vs Predicted)")
            st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 2: ANOMALY INVESTIGATION ---
with tab2:
    if st.session_state.predictions is None:
        st.info("Please train the model first.")
    else:
        df = st.session_state.predictions
        anomalies = df[df['is_anomaly_pred'] == 1]
        
        st.subheader("🚨 Top Detected Anomalies")
        st.write("These sessions have the highest anomaly scores (most deviant behavior).")
        
        # Sort by score descending
        top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(10)
        st.dataframe(top_anomalies[['user_id', 'files_accessed', 'failed_logins', 
                                    'data_outbound_mb', 'cpu_usage', 'file_encryption_rate', 
                                    'anomaly_score']], use_container_width=True)
        
        st.divider()
        st.subheader("Feature Correlation Analysis")
        
        # 3D Scatter Plot
        fig_3d = px.scatter_3d(
            df.sample(min(2000, len(df))), # Sample to keep UI fast
            x='files_accessed', 
            y='file_encryption_rate', 
            z='cpu_usage',
            color='is_anomaly_pred',
            color_discrete_map={0: 'blue', 1: 'red'},
            title="3D View: Files vs Encryption vs CPU",
            opacity=0.7
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        col_scatter1, col_scatter2 = st.columns(2)
        
        with col_scatter1:
            fig_scat1 = px.scatter(
                df, x="files_accessed", y="data_outbound_mb", color="is_anomaly_pred",
                title="Data Exfiltration Patterns",
                color_discrete_map={0: "blue", 1: "red"}
            )
            st.plotly_chart(fig_scat1, use_container_width=True)
            
        with col_scatter2:
            fig_scat2 = px.scatter(
                df, x="failed_logins", y="session_duration", color="is_anomaly_pred",
                title="Brute Force Patterns",
                color_discrete_map={0: "blue", 1: "red"}
            )
            st.plotly_chart(fig_scat2, use_container_width=True)

# --- TAB 3: LIVE SIMULATION ---
with tab3:
    st.subheader("🔴 Real-Time Threat Monitor")
    st.write("Simulate incoming traffic logs and detect threats instantly.")
    
    col_sim_btn, col_sim_stat = st.columns([1, 3])
    
    with col_sim_btn:
        run_sim = st.toggle("Start Monitoring")
    
    if run_sim and st.session_state.model is not None:
        placeholder = st.empty()
        # Simulation loop
        for i in range(100):
            if not run_sim: break
            
            # Generate 1 random log
            new_log = generate_banking_data(1) 
            
            # Preprocess
            processed_log = preprocess_data(new_log)
            
            # Prepare features
            X_new = processed_log[st.session_state.features].values
            X_new_scaled = st.session_state.scaler.transform(X_new)
            
            # Predict
            pred = st.session_state.model.predict(X_new_scaled)
            score = -st.session_state.model.score_samples(X_new_scaled)
            
            is_threat = pred[0] == -1
            
            with placeholder.container():
                # Dynamic Status Card
                if is_threat:
                    st.error(f"⚠️ THREAT DETECTED at {time.strftime('%H:%M:%S')}")
                    st.metric("Anomaly Score", f"{score[0]:.4f}", delta="High Risk", delta_color="inverse")
                else:
                    st.success(f"✅ Normal Activity at {time.strftime('%H:%M:%S')}")
                    st.metric("Anomaly Score", f"{score[0]:.4f}", delta="Low Risk")
                
                st.write("**Incoming Log Details:**")
                st.dataframe(new_log[['user_id', 'files_accessed', 'cpu_usage', 'file_encryption_rate', 'data_outbound_mb']])
                
            time.sleep(1.5)
            
    elif run_sim and st.session_state.model is None:
        st.warning("Please train the model in the Sidebar first!")