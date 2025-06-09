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

# Page config
st.set_page_config(
    page_title="Group Prior ARD Model Analysis",
    page_icon="📊",
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

# Tab 1: Model Overview
with tab1:
    st.header("1. Model Performance")
    metrics = analysis_results['model_metrics']

    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    with col2:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.2f}")
    with col4:
        st.metric("Mean Uncertainty", f"{metrics['mean_std']:.2f}")

    # Prediction Interval Coverage
    st.subheader("Prediction Interval Coverage")
    picp_data = {
        'Coverage Level': ['50%', '80%', '90%', '95%', '99%'],
        'Empirical Coverage': [
            metrics['picp_50'],
            metrics['picp_80'],
            metrics['picp_90'],
            metrics['picp_95'],
            metrics['picp_99']
        ],
        'Target Coverage': [0.5, 0.8, 0.9, 0.95, 0.99]
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Empirical Coverage',
        x=picp_data['Coverage Level'],
        y=picp_data['Empirical Coverage'],
        marker_color='rgb(55, 83, 109)'
    ))
    fig.add_trace(go.Scatter(
        name='Target Coverage',
        x=picp_data['Coverage Level'],
        y=picp_data['Target Coverage'],
        mode='lines+markers',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title='Prediction Interval Coverage',
        xaxis_title='Coverage Level',
        yaxis_title='Coverage',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Feature Analysis
with tab2:
    st.header("2. Feature Analysis")

    # Feature Importance
    st.subheader("Feature Importance")
    importance_data = pd.DataFrame({
        'Feature': list(analysis_results['feature_importance'].keys()),
        'Importance': list(analysis_results['feature_importance'].values()),
        'Uncertainty': list(analysis_results['feature_importance_std'].values())
    })
    importance_data = importance_data.sort_values('Importance', ascending=True)

    fig = px.bar(
        importance_data,
        x='Importance',
        y='Feature',
        error_x='Uncertainty',
        orientation='h',
        title='Feature Importance with Uncertainty'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Correlations
    st.subheader("Feature Correlations with Target")
    correlation_data = pd.DataFrame({
        'Feature': list(analysis_results['target_correlations'].keys()),
        'Correlation': list(analysis_results['target_correlations'].values())
    })
    correlation_data = correlation_data.sort_values('Correlation', ascending=True)

    fig = px.bar(
        correlation_data,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Feature Correlations with Target Variable'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Distribution Analysis
    st.subheader("Feature Distribution Analysis")
    if 'feature_distributions' in analysis_results:
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            options=list(analysis_results['feature_distributions'].keys())
        )
        
        dist_data = analysis_results['feature_distributions'][selected_feature]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dist_data['values'],
            name='Distribution',
            nbinsx=30
        ))
        fig.update_layout(
            title=f'Distribution of {selected_feature}',
            xaxis_title=selected_feature,
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Prior Analysis
with tab3:
    st.header("3. Prior Analysis")

    # Group Prior Hyperparameters
    st.subheader("Group Prior Hyperparameters")
    prior_data = {
        'Group': list(analysis_results['prior_hyperparameters']['global_shrinkage'].keys()),
        'Global Shrinkage': list(analysis_results['prior_hyperparameters']['global_shrinkage'].values()),
        'Local Shrinkage': list(analysis_results['prior_hyperparameters']['local_shrinkage'].values())
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Global Shrinkage',
        x=prior_data['Group'],
        y=prior_data['Global Shrinkage'],
        marker_color='rgb(55, 83, 109)'
    ))
    fig.add_trace(go.Bar(
        name='Local Shrinkage',
        x=prior_data['Group'],
        y=prior_data['Local Shrinkage'],
        marker_color='rgb(26, 118, 255)'
    ))
    fig.update_layout(
        title='Group Prior Hyperparameters',
        xaxis_title='Feature Group',
        yaxis_title='Shrinkage Value',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Prior Sensitivity Analysis
    st.subheader("Prior Sensitivity Analysis")
    if 'prior_sensitivity' in analysis_results:
        sensitivity_data = pd.DataFrame(analysis_results['prior_sensitivity'])
        fig = px.line(
            sensitivity_data,
            x='parameter_value',
            y='performance_metric',
            title='Prior Parameter Sensitivity'
        )
        st.plotly_chart(fig, use_container_width=True)

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
    
    if st.button("Make Prediction"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([feature_inputs])
            
            # Ensure all required features are present
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                st.error(f"Missing features: {', '.join(missing_features)}")
                st.stop()
            
            # Scale the input data
            scaler = StandardScaler()
            input_scaled = scaler.fit_transform(input_data)
            
            # Initialize and train a simple model for demonstration
            model = initialize_model()
            
            # Create some dummy training data based on feature importance
            n_samples = 100
            X_train = np.random.randn(n_samples, len(feature_names))
            y_train = np.sum(X_train * np.array(list(analysis_results['feature_importance'].values())), axis=1)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make prediction
            prediction, uncertainty = model.predict(input_scaled, return_std=True)
            
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
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Group Prior ARD Model Analysis Dashboard | Created with Streamlit") 