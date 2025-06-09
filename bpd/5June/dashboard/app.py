import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import time
from PIL import Image

# Page config
st.set_page_config(
    page_title="Group Prior ARD Model Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Group Prior ARD Model")
    st.markdown("""
    This dashboard analyzes the Group Prior ARD model, which implements different Bayesian priors for different feature groups:
    - Energy features: Horseshoe prior
    - Building features: Hierarchical prior
    - Interaction features: Spike-and-slab prior
    """)
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("- Model Overview")
    st.markdown("- Feature Analysis")
    st.markdown("- Prior Analysis")
    st.markdown("- Predictions")
    
    st.markdown("---")
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader("Upload your data (CSV format)", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            st.session_state['uploaded_data'] = data
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

# Title and Description
st.title("Group Prior ARD Model Analysis")
st.markdown("""
This dashboard provides a comprehensive analysis of the Group Prior ARD model, which implements a novel approach to Bayesian building energy modeling by applying different prior types to different feature groups.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "Model Overview", 
    "Feature Analysis", 
    "Prior Analysis", 
    "Predictions"
])

# Load model results
results_dir = "results_groupprior"
analysis_file = os.path.join(results_dir, "detailed_analysis.json")

if os.path.exists(analysis_file):
    with open(analysis_file, 'r') as f:
        analysis_results = json.load(f)
else:
    st.error("Analysis results not found. Please ensure detailed_analysis.json exists in the results directory.")
    st.stop()

# Initialize a new model for predictions
@st.cache_resource
def initialize_model():
    # Create a Gaussian Process model with similar characteristics
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42
    )
    return model

# Add help tooltips
def add_help_tooltip(text, help_text):
    return f"{text} ℹ️", help_text

# Tab 1: Model Overview
with tab1:
    st.header("1. Model Performance")
    metrics = analysis_results['model_metrics']

    # Add help tooltips to metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(*add_help_tooltip("RMSE", "Root Mean Square Error: Measures the average magnitude of prediction errors"))
    with col2:
        st.metric(*add_help_tooltip("R² Score", "Coefficient of Determination: Measures the proportion of variance explained by the model"))
    with col3:
        st.metric(*add_help_tooltip("MAE", "Mean Absolute Error: Measures the average absolute difference between predictions and actual values"))
    with col4:
        st.metric(*add_help_tooltip("Mean Uncertainty", "Average prediction uncertainty across all samples"))

    # Add residual analysis plot
    st.subheader("Residual Analysis")
    residual_img_path = os.path.join(results_dir, "residual_analysis.png")
    if os.path.exists(residual_img_path):
        image = Image.open(residual_img_path)
        st.image(image, caption="Residual Analysis", use_container_width=True)
    else:
        st.warning("Residual analysis plot not found.")

    # Add calibration plot
    st.subheader("Calibration Plot")
    calibration_img_path = os.path.join(results_dir, "calibration_plot.png")
    if os.path.exists(calibration_img_path):
        image = Image.open(calibration_img_path)
        st.image(image, caption="Calibration Plot", use_container_width=True)
    else:
        st.warning("Calibration plot not found.")

    # Add learning curves plot
    st.subheader("Learning Curves")
    learning_img_path = os.path.join(results_dir, "learning_curves.png")
    if os.path.exists(learning_img_path):
        image = Image.open(learning_img_path)
        st.image(image, caption="Learning Curves", use_container_width=True)
    else:
        st.warning("Learning curves plot not found.")

# Tab 2: Feature Analysis
with tab2:
    st.header("2. Feature Analysis")

    # Feature Importance
    st.subheader("Feature Importance")
    importance_img_path = os.path.join(results_dir, "feature_importance.png")
    if os.path.exists(importance_img_path):
        image = Image.open(importance_img_path)
        st.image(image, caption="Feature Importance", use_container_width=True)
    else:
        st.warning("Feature importance plot not found.")

    # Feature Correlations
    st.subheader("Feature Correlations with Target")
    corr_img_path = os.path.join(results_dir, "importance_vs_correlation.png")
    if os.path.exists(corr_img_path):
        image = Image.open(corr_img_path)
        st.image(image, caption="Feature Importance vs Correlation", use_container_width=True)
    else:
        st.warning("Importance vs Correlation plot not found.")

    # Feature Correlation Matrix
    st.subheader("Feature Correlation Matrix")
    correlation_img_path = os.path.join(results_dir, "correlation_heatmap.png")
    if os.path.exists(correlation_img_path):
        image = Image.open(correlation_img_path)
        st.image(image, caption="Feature Correlation Matrix", use_container_width=True)
    else:
        st.warning("Correlation heatmap not found.")

    # Feature Distribution Analysis (optional: keep or remove if you have a plot)
    st.subheader("Feature Distribution Analysis")
    dist_img_path = os.path.join(results_dir, "uncertainty_distribution.png")
    if os.path.exists(dist_img_path):
        image = Image.open(dist_img_path)
        st.image(image, caption="Uncertainty Distribution", use_container_width=True)
    else:
        st.warning("Uncertainty distribution plot not found.")

# Tab 3: Prior Analysis
with tab3:
    st.header("3. Prior Analysis")

    # Group Prior Hyperparameters (optional: keep code or use image if available)
    st.subheader("Group Prior Hyperparameters")
    group_img_path = os.path.join(results_dir, "group_importance.png")
    if os.path.exists(group_img_path):
        image = Image.open(group_img_path)
        st.image(image, caption="Group Importance", use_container_width=True)
    else:
        st.warning("Group importance plot not found.")

    # Prior Sensitivity Analysis (optional: add if you have a plot)
    st.subheader("Prior Sensitivity Analysis")
    # If you have a plot, display it here

# Tab 4: Predictions
with tab4:
    st.header("4. Make Predictions")
    
    st.markdown("""
    ### Input Building Features
    Please provide the building features below to get a prediction with uncertainty estimates.
    """)
    
    # Create input fields for features
    col1, col2 = st.columns(2)
    feature_inputs = {}
    
    # Get feature names from the analysis results
    feature_names = list(analysis_results['feature_importance'].keys())
    mid_point = len(feature_names) // 2
    
    with col1:
        for feature in feature_names[:mid_point]:
            feature_inputs[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.2f",
                help=f"Enter the value for {feature}"
            )
    
    with col2:
        for feature in feature_names[mid_point:]:
            feature_inputs[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.2f",
                help=f"Enter the value for {feature}"
            )
    
    # Add progress bar for prediction
    if st.button("Make Prediction"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare input data
            status_text.text("Preparing input data...")
            input_data = pd.DataFrame([feature_inputs])
            time.sleep(0.5)  # Add small delay for visual feedback
            progress_bar.progress(25)
            
            # Ensure all required features are present
            status_text.text("Validating features...")
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                st.error(f"Missing features: {', '.join(missing_features)}")
                st.stop()
            time.sleep(0.5)
            progress_bar.progress(50)
            
            # Scale the input data
            status_text.text("Scaling input data...")
            scaler = StandardScaler()
            input_scaled = scaler.fit_transform(input_data)
            time.sleep(0.5)
            progress_bar.progress(75)
            
            # Make prediction
            status_text.text("Making prediction...")
            model = initialize_model()
            prediction, uncertainty = model.predict(input_scaled, return_std=True)
            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.text("Prediction complete!")
            
            # Display results
            st.markdown("### Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Value", f"{prediction[0]:.2f}")
            with col2:
                st.metric("Uncertainty (1σ)", f"{uncertainty[0]:.2f}")
            
            # Display prediction intervals
            st.markdown("### Prediction Intervals")
            intervals = {
                "68%": (prediction[0] - uncertainty[0], prediction[0] + uncertainty[0]),
                "95%": (prediction[0] - 2*uncertainty[0], prediction[0] + 2*uncertainty[0]),
                "99.7%": (prediction[0] - 3*uncertainty[0], prediction[0] + 3*uncertainty[0])
            }
            
            for interval, (lower, upper) in intervals.items():
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{interval} Confidence Interval</h4>
                    <p>Lower bound: {lower:.2f}</p>
                    <p>Upper bound: {upper:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Add download button for predictions
            if st.button("Download Predictions"):
                prediction_df = pd.DataFrame({
                    'Feature': list(feature_inputs.keys()),
                    'Value': list(feature_inputs.values())
                })
                prediction_df['Predicted Value'] = prediction[0]
                prediction_df['Uncertainty'] = uncertainty[0]
                st.markdown(download_dataframe(prediction_df, 'predictions.csv'), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Footer
st.markdown("---")
st.markdown("Group Prior ARD Model Analysis Dashboard | Created with Streamlit")

# Add new functions after the existing imports
def plot_correlation_matrix(correlation_data):
    # Create correlation matrix
    corr_matrix = pd.DataFrame(correlation_data).set_index('Feature')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    return plt

def download_dataframe(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href 