
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("GeoAuPredict 3D Visualization", 
            style={'textAlign': 'center', 'color': '#1f77b4'}),

    html.Div([
        html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'XGBoost', 'value': 'xgb'},
                    {'label': 'LightGBM', 'value': 'lgb'},
                    {'label': 'SVM', 'value': 'svm'},
                    {'label': 'KNN', 'value': 'knn'}
                ],
                value='rf'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Probability Threshold:"),
            dcc.Slider(
                id='threshold-slider',
                min=0.0,
                max=1.0,
                step=0.1,
                value=0.5,
                marks={i/10: str(i/10) for i in range(0, 11)}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),

        html.Div([
            html.Label("Show Uncertainty:"),
            dcc.Checklist(
                id='uncertainty-checkbox',
                options=[{'label': 'Show Uncertainty', 'value': 'show'}],
                value=['show']
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
    ], style={'marginBottom': '20px'}),

    dcc.Graph(id='3d-map'),

    html.Div([
        html.Div([
            html.H3("Model Performance"),
            html.Div(id='performance-metrics')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Exploration Targets"),
            html.Div(id='target-metrics')
        ], style={'width': '50%', 'display': 'inline-block'})
    ])
])

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
            colorscale='YlOrRd',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Probability"),
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
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
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
        html.P(f"AUC Score: {perf['auc']:.3f}"),
        html.P(f"Precision: {perf['precision']:.3f}"),
        html.P(f"Recall: {perf['recall']:.3f}")
    ])

    # Target metrics
    high_priority = len(filtered_data[filtered_data['probability'] >= 0.7])
    medium_priority = len(filtered_data[(filtered_data['probability'] >= 0.5) & 
                                      (filtered_data['probability'] < 0.7)])

    target_metrics = html.Div([
        html.P(f"High Priority: {high_priority}"),
        html.P(f"Medium Priority: {medium_priority}"),
        html.P(f"Total Targets: {len(filtered_data)}"),
        html.P(f"Coverage: {len(filtered_data)/n_points*100:.1f}%")
    ])

    return fig, performance_metrics, target_metrics

if __name__ == '__main__':
    app.run(debug=True, port=8051)
