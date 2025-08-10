import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

# Import plotly with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Please run 'pip install plotly' in your terminal.")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .safe-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .danger-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Heart disease dataset not found. Please ensure 'heart.csv' is in the same directory.")
        return None

@st.cache_resource
def train_models(df):
    """Train multiple models and cache the results"""
    if df is None:
        return None, None, None, None, None, None, None  # Always return 7 values
    
    # Data preprocessing
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'st_slope']
    numerical_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Handle missing values
    df['resting_bp_s'] = df['resting_bp_s'].replace(0, df['resting_bp_s'].mean())
    df['cholesterol'] = df['cholesterol'].replace(0, df['cholesterol'].mean())
    df['oldpeak'] = df['oldpeak'].clip(lower=0)
    
    # One-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Scaling
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=pd.Index(numerical_cols))
    scaled_df = pd.concat([scaled_numerical_df, encoded_df, df['target']], axis=1)
    
    # Feature engineering
    scaled_df['age_chol_interaction'] = scaled_df['age'] * scaled_df['cholesterol']
    scaled_df['sex_heart_rate_interaction'] = scaled_df['sex_1.0'] * scaled_df['max_heart_rate']
    
    # Prepare features and target
    X = scaled_df.drop('target', axis=1)
    y = scaled_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'training_time': training_time
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        trained_models[name] = model
        model_metrics[name] = metrics
    
    return trained_models, model_metrics, encoder, scaler, X.columns, X_test, y_test

def preprocess_input(input_data, encoder, scaler, feature_columns):
    """Preprocess user input for prediction"""
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Categorical and numerical columns
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 'exercise_angina', 'st_slope']
    numerical_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak']
    
    # Encode categorical features
    encoded_features = encoder.transform(df_input[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Scale numerical features
    scaled_numerical = scaler.transform(df_input[numerical_cols])
    scaled_df = pd.DataFrame(scaled_numerical, columns=pd.Index(numerical_cols))
    
    # Combine features
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Add interaction features
    processed_df['age_chol_interaction'] = processed_df['age'] * processed_df['cholesterol']
    processed_df['sex_heart_rate_interaction'] = processed_df['sex_1.0'] * processed_df['max_heart_rate']
    
    # Ensure column order matches training data
    processed_df = processed_df[feature_columns]
    
    return processed_df

def main():
    # Main header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Model Performance", "🔮 Make Prediction", "📈 Data Analysis"]
    )
    
    # Train models (cached)
    trained_models, model_metrics, encoder, scaler, feature_columns, X_test, y_test = train_models(df)
    
    if page == "🏠 Home":
        show_home_page(df)
    
    elif page == "📊 Model Performance":
        show_model_performance(model_metrics, trained_models, X_test, y_test)
    
    elif page == "🔮 Make Prediction":
        show_prediction_page(trained_models, encoder, scaler, feature_columns, model_metrics)
    
    elif page == "📈 Data Analysis":
        show_data_analysis(df)

def show_home_page(df):
    """Display the home page with overview information"""
    st.markdown('<h2 class="sub-header">Welcome to the Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 About This System
        
        This advanced heart disease prediction system uses machine learning to analyze patient symptoms 
        and provide accurate predictions about the likelihood of heart disease. Our system incorporates 
        multiple state-of-the-art algorithms to ensure the highest accuracy possible.
        
        ### 🔬 Key Features
        
        - **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
        - **Real-time Predictions**: Get instant predictions with confidence scores
        - **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score analysis
        - **Interactive Interface**: User-friendly design for easy symptom input
        - **Data Visualization**: Comprehensive analysis of the dataset
        
        ### 📋 Required Symptoms
        
        The system analyzes the following key symptoms and measurements:
        """)
        
        symptoms = [
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
            "Cholesterol Level", "Fasting Blood Sugar", "Resting ECG Results",
            "Maximum Heart Rate", "Exercise-Induced Angina", "ST Depression",
            "ST Slope"
        ]
        
        for i, symptom in enumerate(symptoms, 1):
            st.markdown(f"**{i}.** {symptom}")
    
    with col2:
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset statistics
        total_patients = len(df)
        heart_disease_cases = df['target'].sum()
        healthy_cases = total_patients - heart_disease_cases
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Patients</h3>
            <h2>{total_patients:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Heart Disease Cases</h3>
            <h2>{heart_disease_cases:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Healthy Cases</h3>
            <h2>{healthy_cases:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Navigate to 'Make Prediction'** to input patient symptoms
    2. **Check 'Model Performance'** to see accuracy metrics
    3. **Explore 'Data Analysis'** to understand the dataset
    """)

def show_model_performance(model_metrics, trained_models, X_test, y_test):
    """Display model performance metrics"""
    st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    if model_metrics is None:
        st.error("❌ Model metrics not available. Please ensure models are trained.")
        return
    
    # Performance comparison
    st.markdown("### 🏆 All Models Comparison")
    
    # Create performance DataFrame
    perf_df = pd.DataFrame(model_metrics).T
    perf_df = perf_df.round(4)
    # Update accuracy values for highest and lowest (for extra table only)
    if 'accuracy' in perf_df.columns:
        max_acc = perf_df['accuracy'].max()
        min_acc = perf_df['accuracy'].min()
        max_idx = perf_df['accuracy'].idxmax()
        min_idx = perf_df['accuracy'].idxmin()
    # Display main metrics table (all models, no labels) with all rows visible (not scrollable)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(perf_df, use_container_width=True)
    with col2:
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 32px 16px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
            <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'>🏆 Best Model</div>
            <div style='font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;'>{best_model[0]}</div>
            <div style='font-size: 1.2rem; margin-top: 1rem;'>Accuracy: {best_model[1]['accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    # Extra table for highest and lowest accuracy models
    st.markdown("## Model Accuracy Comparison (Highest & Lowest Accuracy)")
    extra_df = perf_df.loc[[max_idx, min_idx]].copy()
    if str(max_idx) == str(min_idx):
        # Only one model, show only one row
        extra_df = perf_df.loc[[max_idx]].copy()
    if 'accuracy' in extra_df.columns:
        if max_idx in extra_df.index:
            extra_df.loc[max_idx, 'accuracy'] = f"Highest: {max_acc:.2%}"
        if min_idx in extra_df.index:
            extra_df.loc[min_idx, 'accuracy'] = f"Lowest: {min_acc:.2%}"
    st.dataframe(extra_df, use_container_width=True)
    # Performance visualization
    st.markdown("### 📈 Performance Metrics Visualization")
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    models = list(model_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        values = [model_metrics[model][metric] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric.title()),
            row=row, col=col
        )
    fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    # Training time comparison
    st.markdown("### ⏱️ Training Time Comparison")
    training_times = [model_metrics[model]['training_time'] for model in models]
    fig_time = px.bar(
        x=models, 
        y=training_times,
        title="Model Training Time (seconds)",
        labels={'x': 'Model', 'y': 'Training Time (seconds)'}
    )
    fig_time.update_traces(marker_color='lightcoral')
    st.plotly_chart(fig_time, use_container_width=True)
    # Cross-validation scores
    st.markdown("### 🔄 Cross-Validation Results")
    if st.button("Run Cross-Validation Analysis"):
        with st.spinner("Running cross-validation..."):
            cv_results = {}
            for name, model in trained_models.items():
                cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
                cv_results[name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_scores': cv_scores
                }
            # Display CV results
            cv_df = pd.DataFrame({
                'Model': list(cv_results.keys()),
                'Mean CV Score': [cv_results[model]['mean_cv_score'] for model in cv_results.keys()],
                'Std CV Score': [cv_results[model]['std_cv_score'] for model in cv_results.keys()]
            }).round(4)
            st.dataframe(cv_df, use_container_width=True)

def show_prediction_page(trained_models, encoder, scaler, feature_columns, model_metrics):
    """Display the prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Heart Disease Prediction</h2>', unsafe_allow_html=True)
    
    if trained_models is None:
        st.error("❌ Models not available. Please ensure models are trained.")
        return
    
    # Best model for prediction
    best_model_name = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = trained_models[best_model_name]
    
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ Using Model:</strong> {best_model_name} (Accuracy: {model_metrics[best_model_name]['accuracy']:.2%})
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        age = st.slider("Age", min_value=20, max_value=100, value=50, help="Patient's age in years")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Patient's biological sex")
        
        st.markdown("#### Vital Signs")
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, help="Systolic blood pressure at rest")
        cholesterol = st.slider("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=250, help="Serum cholesterol level")
        max_heart_rate = st.slider("Maximum Heart Rate", min_value=60, max_value=202, value=150, help="Maximum heart rate achieved during exercise")
    
    with col2:
        st.markdown("#### Medical Tests")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina", 
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x],
            help="Type of chest pain experienced"
        )
        
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether fasting blood sugar is above 120 mg/dl"
        )
        
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x],
            help="Results of resting electrocardiogram"
        )
        
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether angina is induced by exercise"
        )
        
        oldpeak = st.slider("ST Depression", min_value=0.0, max_value=6.0, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest")
        
        st_slope = st.selectbox(
            "ST Slope",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x],
            help="Slope of the peak exercise ST segment"
        )
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease", type="primary"):
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'resting_bp_s': resting_bp,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fasting_blood_sugar,
            'resting_ecg': resting_ecg,
            'max_heart_rate': max_heart_rate,
            'exercise_angina': exercise_angina,
            'oldpeak': oldpeak,
            'st_slope': st_slope
        }
        
        # Preprocess input
        processed_input = preprocess_input(input_data, encoder, scaler, feature_columns)
        
        # Make prediction with timing
        start_time = time.time()
        prediction = best_model.predict(processed_input)[0]
        prediction_proba = best_model.predict_proba(processed_input)[0][1] if hasattr(best_model, 'predict_proba') else None
        prediction_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-card danger-prediction">
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Heart Disease Detected</h3>
                    <p>Please consult a healthcare professional immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-card safe-prediction">
                    <h2>✅ LOW RISK</h2>
                    <h3>No Heart Disease Detected</h3>
                    <p>Continue maintaining a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if prediction_proba is not None:
                confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence Level</h3>
                    <h2>{confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction Time</h3>
                <h2>{prediction_time:.3f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed probability
        if prediction_proba is not None:
            st.markdown("### 📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability of Heart Disease"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk factors analysis
                st.markdown("#### 🚨 Risk Factors Analysis")
                
                risk_factors = []
                if age > 65:
                    risk_factors.append("Advanced age (>65 years)")
                if resting_bp > 140:
                    risk_factors.append("High blood pressure (>140 mm Hg)")
                if cholesterol > 300:
                    risk_factors.append("High cholesterol (>300 mg/dl)")
                if max_heart_rate < 100:
                    risk_factors.append("Low maximum heart rate (<100)")
                if exercise_angina == 1:
                    risk_factors.append("Exercise-induced angina")
                if oldpeak > 2:
                    risk_factors.append("Significant ST depression (>2)")
                
                if risk_factors:
                    st.markdown("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("✅ No significant risk factors identified")

def show_data_analysis(df):
    """Display data analysis and visualizations"""
    st.markdown('<h2 class="sub-header">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    # Basic statistics
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Heart Disease Cases", df['target'].sum())
    with col3:
        st.metric("Healthy Cases", len(df) - df['target'].sum())
    with col4:
        st.metric("Disease Rate", f"{df['target'].mean():.1%}")
    # Distribution plots
    st.markdown("### 📈 Feature Distributions")
    fig_age = px.histogram(
        df, x='age', color='target',
        title="Age Distribution by Heart Disease Status",
        labels={'age': 'Age', 'target': 'Heart Disease'},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig_age, use_container_width=True)
    fig_scatter = px.scatter(
        df, x='resting_bp_s', y='cholesterol', color='target',
        title="Blood Pressure vs Cholesterol by Heart Disease Status",
        labels={'resting_bp_s': 'Resting Blood Pressure', 'cholesterol': 'Cholesterol'},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("### 🔗 Feature Correlations")
    corr_matrix = df.corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("### 📊 Disease Prevalence by Categories")
    col1, col2 = st.columns(2)
    with col1:
        sex_dist = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
        fig_sex = px.bar(
            x=['Female', 'Male'],
            y=[sex_dist[0], sex_dist[1]],
            title="Heart Disease by Sex",
            labels={'x': 'Sex', 'y': 'Count'}
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    with col2:
        chest_pain_dist = df.groupby(['chest_pain_type', 'target']).size().unstack(fill_value=0)
        y_vals = [chest_pain_dist.loc[i].sum() if i in chest_pain_dist.index else 0 for i in [1,2,3,4]]
        fig_chest = px.bar(
            x=['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'],
            y=y_vals,
            title="Heart Disease by Chest Pain Type",
            labels={'x': 'Chest Pain Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_chest, use_container_width=True)

if __name__ == "__main__":
    main()
