# ARD Model Comparison Dashboard

This interactive dashboard visualizes the comparative analysis between the Adaptive Prior ARD and Feature-Group Adaptive Prior models.

## Features

- Interactive visualizations of model metrics
- Feature importance comparison
- Prior hyperparameter analysis
- Feature interaction strengths
- Model recommendations and future improvements

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

To run the dashboard locally:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Usage

The dashboard is organized into five main sections:

1. **Model Metrics**: Compare performance metrics between both models
2. **Feature Importance**: Visualize feature importance scores
3. **Prior Hyperparameters**: Analyze global and local shrinkage parameters
4. **Feature Interactions**: Explore interaction strengths between features
5. **Recommendations**: Guidelines for model selection and future improvements

## Deployment

To deploy this dashboard online, you can use Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the dashboard

## Requirements

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy
- Matplotlib
- Seaborn 