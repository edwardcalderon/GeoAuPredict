
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import folium
from folium import plugins
import rasterio
from rasterio.plot import show

# Page configuration
st.set_page_config(
    page_title="GeoAuPredict - Spatial Validation Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🗺️ GeoAuPredict Spatial Validation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("### Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost", "LightGBM", "SVM", "KNN"],
    index=0
)

st.sidebar.markdown("### Visualization Options")
show_uncertainty = st.sidebar.checkbox("Show Uncertainty", value=True)
probability_threshold = st.sidebar.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.1)

# Load data (in real app, this would be loaded from files)
@st.cache_data
def load_spatial_results():
    """Load spatial validation results"""
    return {
        'cv_results': {
            'Random Forest': {'mean_score': 0.85, 'std_score': 0.05},
            'XGBoost': {'mean_score': 0.82, 'std_score': 0.06},
            'LightGBM': {'mean_score': 0.80, 'std_score': 0.07},
            'SVM': {'mean_score': 0.78, 'std_score': 0.08},
            'KNN': {'mean_score': 0.75, 'std_score': 0.09}
        },
        'precision_at_k': {
            'Random Forest': {10: 0.8, 20: 0.75, 50: 0.7, 100: 0.65},
            'XGBoost': {10: 0.75, 20: 0.7, 50: 0.65, 100: 0.6},
            'LightGBM': {10: 0.7, 20: 0.65, 50: 0.6, 100: 0.55},
            'SVM': {10: 0.65, 20: 0.6, 50: 0.55, 100: 0.5},
            'KNN': {10: 0.6, 20: 0.55, 50: 0.5, 100: 0.45}
        }
    }

# Load results
results = load_spatial_results()

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
    st.metric(
        "CV AUC Score",
        f"{results['cv_results'][selected_model]['mean_score']:.3f}",
        f"±{results['cv_results'][selected_model]['std_score']:.3f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
    st.metric(
        "Precision@10",
        f"{results['precision_at_k'][selected_model][10]:.3f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
    st.metric(
        "Precision@50",
        f"{results['precision_at_k'][selected_model][50]:.3f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "High Priority Targets",
        "15",
        "5 new"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Model Comparison", "🎯 Exploration Targets", "📈 Analytics"])

with tab1:
    st.subheader("Interactive Probability Map")

    # Create sample map data
    np.random.seed(42)
    n_points = 1000
    lat_min, lat_max = 4.3, 12.5
    lon_min, lon_max = -79.0, -66.8

    map_data = pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, n_points),
        'lon': np.random.uniform(lon_min, lon_max, n_points),
        'probability': np.random.beta(2, 5, n_points),
        'uncertainty': np.random.uniform(0, 0.3, n_points)
    })

    # Filter by threshold
    filtered_data = map_data[map_data['probability'] >= probability_threshold]

    # Create map
    fig = px.scatter_mapbox(
        filtered_data,
        lat='lat',
        lon='lon',
        color='probability',
        size='uncertainty' if show_uncertainty else None,
        hover_data=['probability', 'uncertainty'],
        color_continuous_scale='YlOrRd',
        mapbox_style='open-street-map',
        zoom=6,
        center={'lat': 8.5, 'lon': -73.0},
        title=f'Gold Probability Map - {selected_model}'
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Map statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Points", len(map_data))
    with col2:
        st.metric("Above Threshold", len(filtered_data))
    with col3:
        st.metric("Coverage %", f"{len(filtered_data)/len(map_data)*100:.1f}%")

with tab2:
    st.subheader("Model Performance Comparison")

    # CV Scores comparison
    models = list(results['cv_results'].keys())
    cv_scores = [results['cv_results'][model]['mean_score'] for model in models]
    cv_stds = [results['cv_results'][model]['std_score'] for model in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=cv_scores,
        error_y=dict(type='data', array=cv_stds),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ))

    fig.update_layout(
        title="Cross-Validation AUC Scores",
        xaxis_title="Model",
        yaxis_title="AUC Score",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Precision@k comparison
    k_values = [10, 20, 50, 100]
    precision_data = []

    for model in models:
        precision_data.append([results['precision_at_k'][model][k] for k in k_values])

    fig2 = go.Figure()
    for i, model in enumerate(models):
        fig2.add_trace(go.Scatter(
            x=k_values,
            y=precision_data[i],
            mode='lines+markers',
            name=model,
            line=dict(width=3)
        ))

    fig2.update_layout(
        title="Precision@k Comparison",
        xaxis_title="k",
        yaxis_title="Precision",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Exploration Target Analysis")

    # High priority targets
    high_priority = filtered_data[filtered_data['probability'] >= 0.7]
    medium_priority = filtered_data[(filtered_data['probability'] >= 0.5) & 
                                   (filtered_data['probability'] < 0.7)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### High Priority Targets (>0.7)")
        st.metric("Count", len(high_priority))
        st.metric("Avg Probability", f"{high_priority['probability'].mean():.3f}")
        st.metric("Avg Uncertainty", f"{high_priority['uncertainty'].mean():.3f}")

    with col2:
        st.markdown("### Medium Priority Targets (0.5-0.7)")
        st.metric("Count", len(medium_priority))
        st.metric("Avg Probability", f"{medium_priority['probability'].mean():.3f}")
        st.metric("Avg Uncertainty", f"{medium_priority['uncertainty'].mean():.3f}")

    # Target distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered_data['probability'],
        nbinsx=20,
        name='Probability Distribution'
    ))

    fig.update_layout(
        title="Target Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Count",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Advanced Analytics")

    # Spatial clustering analysis
    st.markdown("### Spatial Clustering Analysis")

    # Calculate spatial statistics
    spatial_stats = {
        'Total Area': '1,141,748 km²',
        'Target Area': f'{len(filtered_data) * 4:.0f} km²',
        'Coverage': f'{len(filtered_data) * 4 / 1141748 * 100:.3f}%',
        'Density': f'{len(filtered_data) / 1000:.2f} targets/km²'
    }

    for stat, value in spatial_stats.items():
        st.metric(stat, value)

    # Uncertainty analysis
    st.markdown("### Uncertainty Analysis")

    uncertainty_stats = {
        'Mean Uncertainty': f"{filtered_data['uncertainty'].mean():.3f}",
        'Std Uncertainty': f"{filtered_data['uncertainty'].std():.3f}",
        'High Uncertainty (>0.2)': f"{len(filtered_data[filtered_data['uncertainty'] > 0.2])}",
        'Low Uncertainty (<0.1)': f"{len(filtered_data[filtered_data['uncertainty'] < 0.1])}"
    }

    for stat, value in uncertainty_stats.items():
        st.metric(stat, value)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>GeoAuPredict Spatial Validation Dashboard | 
    <a href='https://github.com/your-repo/GeoAuPredict' target='_blank'>GitHub</a> | 
    <a href='#'>Documentation</a></p>
</div>
""", unsafe_allow_html=True)
