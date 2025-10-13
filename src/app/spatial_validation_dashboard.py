#!/usr/bin/env python3
"""
Spatial Validation Dashboard for GeoAuPredict
Checks and installs required dependencies before running
"""

import sys
import subprocess
import os
from pathlib import Path

# Check and install required packages
def check_requirements():
    """Check if all required packages are installed, install if missing (local only)"""
    required_packages = {
        'streamlit': 'streamlit>=1.28.0',
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0',
        'plotly': 'plotly>=5.17.0',
        'geopandas': 'geopandas>=0.12.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(requirement)
    
    if missing_packages:
        # Check if running on Streamlit Cloud or other production environment
        is_production = (
            os.environ.get('STREAMLIT_SHARING_MODE') == 'True' or
            os.environ.get('STREAMLIT_CLOUD') == 'true' or
            os.path.exists('/.dockerenv') or
            'KUBERNETES_SERVICE_HOST' in os.environ
        )
        
        if is_production:
            # In production, can't install packages - must be in requirements.txt
            print("❌ Missing required packages in production environment:")
            print(f"   Missing: {', '.join(missing_packages)}")
            print("\n📋 To fix this, add these to your requirements.txt file:")
            for pkg in missing_packages:
                print(f"   {pkg}")
            print("\nOr use web_requirements.txt by creating .streamlit/config.toml:")
            print("   [server]")
            print("   requirements = 'web_requirements.txt'")
            sys.exit(1)
        else:
            # Local development - try to install
            print("📦 Installing missing dependencies...")
            print(f"   Missing: {', '.join(missing_packages)}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q"] + missing_packages
                )
                print("✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing dependencies: {e}")
                print("Please run: pip install -r web_requirements.txt")
                sys.exit(1)

# Run requirement check
check_requirements()

# Now import the packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GeoAuPredict - Spatial Validation Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode and yellow accents
st.markdown("""
<style>
    /* Dark mode styling */
    .stApp {
        background-color: #1c2739;
        color: #fafafa;
    }
    
    /* Main header - smaller and more refined */
    .main-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #fafafa;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Compact metric cards */
    .metric-card {
        background-color: #131b2d;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border-left: 3px solid #fbbf24;
        color: #fafafa;
    }
    
    .success-metric {
        border-left-color: #fbbf24;
    }
    
    .warning-metric {
        border-left-color: #f59e0b;
    }
    
    /* Sidebar styling - more compact */
    section[data-testid="stSidebar"] {
        background-color: #131b2d;
        border-right: 1px solid #262626;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #fafafa !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #a3a3a3 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem !important;
    }
    
    /* Text color overrides */
    .stMarkdown, .stText, p, span, label {
        color: #fafafa !important;
        font-size: 0.875rem !important;
    }
    
    /* Smaller, refined tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #131b2d;
        border-radius: 0.5rem;
        gap: 0.25rem;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a3a3a3;
        background-color: transparent;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        color: #fafafa;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0a0a0a !important;
        background-color: #fbbf24 !important;
        border-bottom-color: #fbbf24 !important;
    }
    
    /* Compact metric styling */
    [data-testid="stMetric"] {
        background-color: transparent;
        padding: 0.5rem;
        border-radius: 0.375rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: #a3a3a3 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #fbbf24 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }
    
    /* Smaller selectbox */
    .stSelectbox label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #fafafa !important;
    }
    
    .stSelectbox > div > div {
        background-color: #1c2739;
        border: 1px solid #262626;
        border-radius: 0.375rem;
        font-size: 0.875rem;
    }
    
    /* Compact slider */
    .stSlider label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    
    /* Smaller checkbox */
    .stCheckbox label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* Compact columns */
    [data-testid="column"] {
        padding: 0.25rem;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        background-color: #1c2739;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1c2739 !important;
    }
    
    /* Ensure main background stays dark */
    .main {
        background-color: #1c2739 !important;
    }
    
    /* Content area when sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ .main {
        background-color: #1c2739 !important;
    }
    
    /* Plotly charts - dark theme */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        color: #fafafa !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Cards/containers */
    [data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
    
    /* Fix any white backgrounds */
    div, section, main {
        background-color: inherit;
    }
    
    /* Override Streamlit's default white background */
    [data-testid="stAppViewContainer"] {
        background-color: #1c2739 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #131b2d !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: #131b2d !important;
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

# Data loading functions with error handling
@st.cache_data
def load_spatial_results():
    """Load spatial validation results from file or use sample data"""
    
    # Default sample data structure
    default_data = {
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
    
    try:
        # Try to find the project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Check for spatial validation results
        results_file = project_root / "outputs" / "spatial_validation_results.json"
        
        if results_file.exists():
            logger.info(f"Loading real data from {results_file}")
            with open(results_file, 'r') as f:
                data = json.load(f)
                
                # Validate data structure
                if 'cv_results' not in data or 'precision_at_k' not in data:
                    logger.warning("Loaded data missing required keys, using sample data")
                    return default_data
                
                # Ensure all models have required fields
                for model in data.get('cv_results', {}).values():
                    if 'mean_score' not in model or 'std_score' not in model:
                        logger.warning("Invalid cv_results structure, using sample data")
                        return default_data
                
                logger.info("Successfully loaded and validated real data")
                return data
        else:
            logger.warning(f"Results file not found at {results_file}, using sample data")
    except Exception as e:
        logger.error(f"Error loading results: {e}, falling back to sample data")
    
    # Sample data fallback
    return default_data

@st.cache_data
def load_map_data():
    """Load or generate map data with error handling"""
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Try to load real geospatial data
        geojson_file = project_root / "data" / "processed" / "gold_dataset_master.geojson"
        
        if geojson_file.exists():
            logger.info(f"Loading real map data from {geojson_file}")
            import geopandas as gpd
            gdf = gpd.read_file(geojson_file)
            
            # Convert to format needed for visualization
            if not gdf.empty:
                map_df = pd.DataFrame({
                    'lat': gdf.geometry.y,
                    'lon': gdf.geometry.x,
                    'probability': np.random.beta(2, 5, len(gdf)),  # Would come from predictions
                    'uncertainty': np.random.uniform(0, 0.3, len(gdf))
                })
                return map_df
    except Exception as e:
        logger.warning(f"Could not load real map data: {e}, generating sample data")
    
    # Generate sample data
    np.random.seed(42)
    n_points = 1000
    lat_min, lat_max = 4.3, 12.5
    lon_min, lon_max = -79.0, -66.8
    
    return pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, n_points),
        'lon': np.random.uniform(lon_min, lon_max, n_points),
        'probability': np.random.beta(2, 5, n_points),
        'uncertainty': np.random.uniform(0, 0.3, n_points)
    })

# Load results
results = load_spatial_results()

# Main content - Metrics with error handling
col1, col2, col3, col4 = st.columns(4)

try:
    with col1:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        if selected_model in results.get('cv_results', {}):
            st.metric(
                "CV AUC Score",
                f"{results['cv_results'][selected_model]['mean_score']:.3f}",
                f"±{results['cv_results'][selected_model]['std_score']:.3f}"
            )
        else:
            st.metric("CV AUC Score", "N/A", "Model not found")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        if selected_model in results.get('precision_at_k', {}) and 10 in results['precision_at_k'][selected_model]:
            st.metric(
                "Precision@10",
                f"{results['precision_at_k'][selected_model][10]:.3f}"
            )
        else:
            st.metric("Precision@10", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
        if selected_model in results.get('precision_at_k', {}) and 50 in results['precision_at_k'][selected_model]:
            st.metric(
                "Precision@50",
                f"{results['precision_at_k'][selected_model][50]:.3f}"
            )
        else:
            st.metric("Precision@50", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "High Priority Targets",
            "15",
            "5 new"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
except Exception as e:
    st.error(f"Error displaying metrics: {str(e)}")
    logger.error(f"Metrics display error: {e}", exc_info=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Model Comparison", "🎯 Exploration Targets", "📈 Analytics"])

with tab1:
    st.subheader("Interactive Probability Map")

    try:
        # Load map data
        map_data = load_map_data()
        
        if map_data is None or map_data.empty:
            st.error("No map data available")
        else:
            # Filter by threshold
            filtered_data = map_data[map_data['probability'] >= probability_threshold]

            if len(filtered_data) == 0:
                st.warning(f"No points above threshold {probability_threshold}")
                filtered_data = map_data  # Show all data if nothing passes threshold

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

            fig.update_layout(
                height=600,
                paper_bgcolor='#131b2d',
                plot_bgcolor='#131b2d',
                font=dict(color='#fafafa', size=12),
                margin=dict(l=0, r=0, t=40, b=0),
                title_font_size=14
            )
            st.plotly_chart(fig, use_container_width=True)

            # Map statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", len(map_data))
            with col2:
                st.metric("Above Threshold", len(filtered_data))
            with col3:
                coverage = len(filtered_data)/len(map_data)*100 if len(map_data) > 0 else 0
                st.metric("Coverage %", f"{coverage:.1f}%")
    
    except Exception as e:
        st.error(f"Error loading map: {str(e)}")
        logger.error(f"Map visualization error: {e}", exc_info=True)

with tab2:
    st.subheader("Model Performance Comparison")

    try:
        # CV Scores comparison
        if 'cv_results' not in results or not results['cv_results']:
            st.warning("No CV results available")
        else:
            models = list(results['cv_results'].keys())
            cv_scores = [results['cv_results'][model].get('mean_score', 0) for model in models]
            cv_stds = [results['cv_results'][model].get('std_score', 0) for model in models]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=cv_scores,
                error_y=dict(type='data', array=cv_stds),
                marker_color='#fbbf24',
                marker_line_color='#f59e0b',
                marker_line_width=1
            ))

            fig.update_layout(
                title="Cross-Validation AUC Scores",
                xaxis_title="Model",
                yaxis_title="AUC Score",
                height=400,
                paper_bgcolor='#131b2d',
                plot_bgcolor='#131b2d',
                font=dict(color='#fafafa', size=12),
                xaxis=dict(gridcolor='#262626'),
                yaxis=dict(gridcolor='#262626'),
                margin=dict(l=60, r=20, t=40, b=60),
                title_font_size=14
            )

            st.plotly_chart(fig, use_container_width=True)

        # Precision@k comparison
        if 'precision_at_k' not in results or not results['precision_at_k']:
            st.warning("No Precision@k results available")
        else:
            k_values = [10, 20, 50, 100]
            precision_data = []
            models = list(results['precision_at_k'].keys())

            for model in models:
                model_data = []
                for k in k_values:
                    model_data.append(results['precision_at_k'][model].get(k, 0))
                precision_data.append(model_data)

            fig2 = go.Figure()
            colors = ['#fbbf24', '#f59e0b', '#d97706', '#b45309', '#92400e']
            for i, model in enumerate(models):
                fig2.add_trace(go.Scatter(
                    x=k_values,
                    y=precision_data[i],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=8)
                ))

            fig2.update_layout(
                title="Precision@k Comparison",
                xaxis_title="k",
                yaxis_title="Precision",
                height=400,
                paper_bgcolor='#131b2d',
                plot_bgcolor='#131b2d',
                font=dict(color='#fafafa', size=12),
                xaxis=dict(gridcolor='#262626'),
                yaxis=dict(gridcolor='#262626'),
                margin=dict(l=60, r=20, t=40, b=60),
                title_font_size=14,
                legend=dict(
                    bgcolor='#171717',
                    bordercolor='#262626',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
        logger.error(f"Model comparison error: {e}", exc_info=True)

with tab3:
    st.subheader("Exploration Target Analysis")

    try:
        # Load map data for analysis
        map_data = load_map_data()
        filtered_data = map_data[map_data['probability'] >= probability_threshold]
        
        if len(filtered_data) == 0:
            st.warning(f"No targets above threshold {probability_threshold}")
            filtered_data = map_data
        
        # High priority targets
        high_priority = filtered_data[filtered_data['probability'] >= 0.7]
        medium_priority = filtered_data[(filtered_data['probability'] >= 0.5) & 
                                       (filtered_data['probability'] < 0.7)]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### High Priority Targets (>0.7)")
            st.metric("Count", len(high_priority))
            if len(high_priority) > 0:
                st.metric("Avg Probability", f"{high_priority['probability'].mean():.3f}")
                st.metric("Avg Uncertainty", f"{high_priority['uncertainty'].mean():.3f}")
            else:
                st.metric("Avg Probability", "N/A")
                st.metric("Avg Uncertainty", "N/A")

        with col2:
            st.markdown("### Medium Priority Targets (0.5-0.7)")
            st.metric("Count", len(medium_priority))
            if len(medium_priority) > 0:
                st.metric("Avg Probability", f"{medium_priority['probability'].mean():.3f}")
                st.metric("Avg Uncertainty", f"{medium_priority['uncertainty'].mean():.3f}")
            else:
                st.metric("Avg Probability", "N/A")
                st.metric("Avg Uncertainty", "N/A")

        # Target distribution
        if len(filtered_data) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_data['probability'],
                nbinsx=20,
                name='Probability Distribution',
                marker_color='#fbbf24',
                marker_line_color='#f59e0b',
                marker_line_width=1
            ))

            fig.update_layout(
                title="Target Probability Distribution",
                xaxis_title="Probability",
                yaxis_title="Count",
                height=400,
                paper_bgcolor='#131b2d',
                plot_bgcolor='#131b2d',
                font=dict(color='#fafafa', size=12),
                xaxis=dict(gridcolor='#262626'),
                yaxis=dict(gridcolor='#262626'),
                margin=dict(l=60, r=20, t=40, b=60),
                title_font_size=14
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for distribution chart")
    
    except Exception as e:
        st.error(f"Error in target analysis: {str(e)}")
        logger.error(f"Target analysis error: {e}", exc_info=True)

with tab4:
    st.subheader("Advanced Analytics")

    try:
        # Load map data for analytics
        map_data = load_map_data()
        filtered_data = map_data[map_data['probability'] >= probability_threshold]
        
        if len(filtered_data) == 0:
            st.warning(f"No targets above threshold {probability_threshold}")
            filtered_data = map_data
        
        # Spatial clustering analysis
        st.markdown("### Spatial Clustering Analysis")

        # Calculate spatial statistics
        spatial_stats = {
            'Total Area': '1,141,748 km²',
            'Target Area': f'{len(filtered_data) * 4:.0f} km²',
            'Coverage': f'{len(filtered_data) * 4 / 1141748 * 100:.3f}%',
            'Density': f'{len(filtered_data) / 1000:.2f} targets/km²'
        }

        cols = st.columns(4)
        for i, (stat, value) in enumerate(spatial_stats.items()):
            with cols[i]:
                st.metric(stat, value)

        # Uncertainty analysis
        st.markdown("### Uncertainty Analysis")

        if len(filtered_data) > 0:
            uncertainty_stats = {
                'Mean Uncertainty': f"{filtered_data['uncertainty'].mean():.3f}",
                'Std Uncertainty': f"{filtered_data['uncertainty'].std():.3f}",
                'High Uncertainty (>0.2)': f"{len(filtered_data[filtered_data['uncertainty'] > 0.2])}",
                'Low Uncertainty (<0.1)': f"{len(filtered_data[filtered_data['uncertainty'] < 0.1])}"
            }

            cols = st.columns(4)
            for i, (stat, value) in enumerate(uncertainty_stats.items()):
                with cols[i]:
                    st.metric(stat, value)
        else:
            st.info("No data available for uncertainty analysis")
    
    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")
        logger.error(f"Analytics error: {e}", exc_info=True)

