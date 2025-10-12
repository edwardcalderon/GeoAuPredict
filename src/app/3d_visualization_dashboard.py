#!/usr/bin/env python3
"""
3D Visualization Dashboard for GeoAuPredict
Checks and installs required dependencies before running
"""

import sys
import subprocess
import os

# Check and install required packages
def check_requirements():
    """Check if all required packages are installed, install if missing (local only)"""
    required_packages = {
        'dash': 'dash>=2.14.0',
        'plotly': 'plotly>=5.17.0',
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(requirement)
    
    if missing_packages:
        # Check if running in production environment
        is_production = (
            os.environ.get('STREAMLIT_SHARING_MODE') == 'True' or
            os.environ.get('STREAMLIT_CLOUD') == 'true' or
            os.path.exists('/.dockerenv') or
            'KUBERNETES_SERVICE_HOST' in os.environ or
            os.environ.get('RENDER') == 'true' or
            os.environ.get('HEROKU') == 'true'
        )
        
        if is_production:
            # In production, can't install packages - must be in requirements.txt
            print("❌ Missing required packages in production environment:")
            print(f"   Missing: {', '.join(missing_packages)}")
            print("\n📋 To fix this, add these to your requirements.txt or web_requirements.txt:")
            for pkg in missing_packages:
                print(f"   {pkg}")
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
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

# Initialize Dash app
app = dash.Dash(__name__)

# Define theme colors to match Next.js app (dark mode with blue tint)
THEME_COLORS = {
    'background': '#1c2739',  # Dark background with blue tint
    'foreground': '#fafafa',   # Light text
    'primary': '#fafafa',      # Light primary
    'secondary': '#131b2d',    # Dark secondary with blue tint
    'muted': '#262626',        # Dark muted
    'border': '#262626',       # Dark border
    'accent': '#fbbf24',       # Yellow accent (gold)
    'card': '#1c2739',         # Dark card
    'card_secondary': '#131b2d',  # Slightly lighter card with blue tint
    'muted_foreground': '#a3a3a3',  # Muted text
    'highlight': '#fbbf24',    # Yellow highlight
    'chart_yellow': '#fbbf24', # Gold/Yellow for charts
    'chart_amber': '#f59e0b'   # Amber for charts
}

# App layout
app.layout = html.Div(
    style={
        'backgroundColor': THEME_COLORS['background'],
        'minHeight': '100vh',
        'padding': '2rem',
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial'
    },
    children=[
        html.H1(
            "GeoAuPredict 3D Visualization",
            style={
                'textAlign': 'center',
                'color': THEME_COLORS['foreground'],
                'marginBottom': '2rem',
                'fontSize': '2.5rem',
                'fontWeight': 'bold'
            }
        ),

        html.Div([
            html.Div([
                html.Label(
                    "Select Model:",
                    style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'display': 'block'}
                ),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'XGBoost', 'value': 'xgb'},
                        {'label': 'LightGBM', 'value': 'lgb'},
                        {'label': 'SVM', 'value': 'svm'},
                        {'label': 'KNN', 'value': 'knn'}
                    ],
                    value='rf',
                    style={
                        'backgroundColor': THEME_COLORS['background'],
                        'color': THEME_COLORS['foreground']
                    }
                )
            ], style={'width': '30%', 'display': 'inline-block', 'paddingRight': '2rem'}),

            html.Div([
                html.Label(
                    "Probability Threshold:",
                    style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'display': 'block'}
                ),
                dcc.Slider(
                    id='threshold-slider',
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    value=0.5,
                    marks={i/10: {'label': str(i/10), 'style': {'color': THEME_COLORS['muted_foreground']}} 
                           for i in range(0, 11)}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'paddingRight': '2rem'}),

            html.Div([
                html.Label(
                    "Show Uncertainty:",
                    style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'display': 'block'}
                ),
                dcc.Checklist(
                    id='uncertainty-checkbox',
                    options=[{'label': ' Show Uncertainty', 'value': 'show'}],
                    value=['show'],
                    style={'color': THEME_COLORS['foreground']}
                )
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '2rem'}),

        dcc.Graph(id='3d-map'),

        html.Div([
            html.Div([
                html.H3(
                    "Model Performance",
                    style={'color': THEME_COLORS['foreground'], 'marginBottom': '1rem'}
                ),
                html.Div(
                    id='performance-metrics',
                    style={
                        'backgroundColor': THEME_COLORS['card_secondary'],
                        'padding': '1.5rem',
                        'borderRadius': '0.5rem',
                        'border': f'1px solid {THEME_COLORS["border"]}'
                    }
                )
            ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),

            html.Div([
                html.H3(
                    "Exploration Targets",
                    style={'color': THEME_COLORS['foreground'], 'marginBottom': '1rem'}
                ),
                html.Div(
                    id='target-metrics',
                    style={
                        'backgroundColor': THEME_COLORS['card_secondary'],
                        'padding': '1.5rem',
                        'borderRadius': '0.5rem',
                        'border': f'1px solid {THEME_COLORS["border"]}'
                    }
                )
            ], style={'width': '48%', 'display': 'inline-block', 'paddingLeft': '2%'})
        ])
    ]
)

@callback(
    [Output('3d-map', 'figure'),
     Output('performance-metrics', 'children'),
     Output('target-metrics', 'children')],
    [Input('model-dropdown', 'value'),
     Input('threshold-slider', 'value'),
     Input('uncertainty-checkbox', 'value')]
)
def update_visualization(selected_model, threshold, uncertainty_options):
    # Generate sample 3D data
    np.random.seed(42)
    n_points = 500

    # Colombia coordinates
    lat_min, lat_max = 4.3, 12.5
    lon_min, lon_max = -79.0, -66.8
    elev_min, elev_max = 0, 3000

    data = pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, n_points),
        'lon': np.random.uniform(lon_min, lon_max, n_points),
        'elev': np.random.uniform(elev_min, elev_max, n_points),
        'probability': np.random.beta(2, 5, n_points),
        'uncertainty': np.random.uniform(0, 0.3, n_points)
    })

    # Filter by threshold
    filtered_data = data[data['probability'] >= threshold]

    # Create 3D scatter plot
    fig = go.Figure()

    # Add probability points
    fig.add_trace(go.Scatter3d(
        x=filtered_data['lon'],
        y=filtered_data['lat'],
        z=filtered_data['elev'],
        mode='markers',
        marker=dict(
            size=5,
            color=filtered_data['probability'],
            colorscale=[[0, '#262626'], [0.3, '#78350f'], [0.5, '#f59e0b'], [0.7, '#fbbf24'], [1, '#fef3c7']],
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Probability",
                tickfont=dict(color=THEME_COLORS['foreground']),
                title_font=dict(color=THEME_COLORS['foreground'])
            ),
            opacity=0.8
        ),
        text=[f'P: {p:.3f}<br>Uncertainty: {u:.3f}' 
              for p, u in zip(filtered_data['probability'], filtered_data['uncertainty'])],
        hovertemplate='<b>Longitude:</b> %{x}<br><b>Latitude:</b> %{y}<br><b>Elevation:</b> %{z}m<br>%{text}<extra></extra>',
        name='Probability'
    ))

    # Add uncertainty if selected
    if 'show' in uncertainty_options:
        fig.add_trace(go.Scatter3d(
            x=filtered_data['lon'],
            y=filtered_data['lat'],
            z=filtered_data['elev'],
            mode='markers',
            marker=dict(
                size=3,
                color=filtered_data['uncertainty'],
                colorscale='Greys',
                cmin=0,
                cmax=0.5,
                opacity=0.3
            ),
            name='Uncertainty',
            visible='legendonly'
        ))

    fig.update_layout(
        title=f'3D Gold Probability Map - {selected_model.upper()}',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)',
            xaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                zerolinecolor=THEME_COLORS['border'],
                color=THEME_COLORS['foreground']
            ),
            yaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                zerolinecolor=THEME_COLORS['border'],
                color=THEME_COLORS['foreground']
            ),
            zaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                zerolinecolor=THEME_COLORS['border'],
                color=THEME_COLORS['foreground']
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        paper_bgcolor=THEME_COLORS['background'],
        plot_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['foreground']),
        title_font=dict(color=THEME_COLORS['foreground'])
    )

    # Performance metrics
    model_performance = {
        'rf': {'auc': 0.85, 'precision': 0.78, 'recall': 0.82},
        'xgb': {'auc': 0.82, 'precision': 0.75, 'recall': 0.79},
        'lgb': {'auc': 0.80, 'precision': 0.73, 'recall': 0.77},
        'svm': {'auc': 0.78, 'precision': 0.70, 'recall': 0.75},
        'knn': {'auc': 0.75, 'precision': 0.68, 'recall': 0.72}
    }

    perf = model_performance[selected_model]
    performance_metrics = html.Div([
        html.P(
            f"AUC Score: {perf['auc']:.3f}",
            style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}
        ),
        html.P(
            f"Precision: {perf['precision']:.3f}",
            style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}
        ),
        html.P(
            f"Recall: {perf['recall']:.3f}",
            style={'color': THEME_COLORS['foreground'], 'fontSize': '1.1rem'}
        )
    ])

    # Target metrics
    high_priority = len(filtered_data[filtered_data['probability'] >= 0.7])
    medium_priority = len(filtered_data[(filtered_data['probability'] >= 0.5) & 
                                      (filtered_data['probability'] < 0.7)])

    target_metrics = html.Div([
        html.P(
            f"High Priority: {high_priority}",
            style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}
        ),
        html.P(
            f"Medium Priority: {medium_priority}",
            style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}
        ),
        html.P(
            f"Total Targets: {len(filtered_data)}",
            style={'color': THEME_COLORS['foreground'], 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}
        ),
        html.P(
            f"Coverage: {len(filtered_data)/n_points*100:.1f}%",
            style={'color': THEME_COLORS['foreground'], 'fontSize': '1.1rem'}
        )
    ])

    return fig, performance_metrics, target_metrics

if __name__ == '__main__':
    app.run(debug=True, port=8050)
