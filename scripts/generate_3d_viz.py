#!/usr/bin/env python3
"""
Generate static interactive 3D visualization for GitHub Pages deployment
Pre-computes all model/threshold combinations for client-side interactivity
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Theme colors matching Next.js app
THEME_COLORS = {
    'background': '#0f172a',  # slate-900
    'foreground': '#f8fafc',  # slate-50
    'accent': '#fbbf24',      # yellow-400
    'border': '#334155',      # slate-700
}

def generate_sample_data():
    """Generate sample prediction data"""
    np.random.seed(42)
    n_points = 500
    
    # Colombia coordinates
    lat_min, lat_max = 4.3, 12.5
    lon_min, lon_max = -79.0, -66.8
    elev_min, elev_max = 0, 3000
    
    return pd.DataFrame({
        'lat': np.random.uniform(lat_min, lat_max, n_points),
        'lon': np.random.uniform(lon_min, lon_max, n_points),
        'elev': np.random.uniform(elev_min, elev_max, n_points),
        'probability': np.random.beta(2, 5, n_points),
        'uncertainty': np.random.uniform(0, 0.3, n_points)
    })

def create_3d_plot(data, model='rf', threshold=0.5, show_uncertainty=True):
    """Create interactive 3D scatter plot"""
    
    # Filter by threshold
    filtered_data = data[data['probability'] >= threshold].copy()
    
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
            colorscale=[
                [0, '#262626'],
                [0.3, '#78350f'],
                [0.5, '#f59e0b'],
                [0.7, '#fbbf24'],
                [1, '#fef3c7']
            ],
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Probability",
                tickfont=dict(color=THEME_COLORS['foreground']),
                title_font=dict(color=THEME_COLORS['foreground']),
                bgcolor='rgba(0,0,0,0.5)'
            ),
            opacity=0.8
        ),
        text=[f'P: {p:.3f}<br>U: {u:.3f}' 
              for p, u in zip(filtered_data['probability'], filtered_data['uncertainty'])],
        hovertemplate='<b>Lon:</b> %{x:.2f}<br><b>Lat:</b> %{y:.2f}<br><b>Elev:</b> %{z:.0f}m<br>%{text}<extra></extra>',
        name='Probability'
    ))
    
    # Add uncertainty layer (initially hidden)
    if show_uncertainty:
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
            visible='legendonly'  # Hidden by default, toggle in legend
        ))
    
    # Layout configuration
    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'lgb': 'LightGBM',
        'svm': 'SVM',
        'knn': 'KNN'
    }
    
    fig.update_layout(
        title=dict(
            text=f'3D Gold Probability Map - {model_names.get(model, model)}',
            font=dict(color=THEME_COLORS['foreground'], size=20)
        ),
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)',
            xaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                color=THEME_COLORS['foreground']
            ),
            yaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                color=THEME_COLORS['foreground']
            ),
            zaxis=dict(
                backgroundcolor=THEME_COLORS['background'],
                gridcolor=THEME_COLORS['border'],
                showbackground=True,
                color=THEME_COLORS['foreground']
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=None,  # Responsive
        height=600,
        paper_bgcolor=THEME_COLORS['background'],
        plot_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['foreground']),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color=THEME_COLORS['foreground'])
        )
    )
    
    return fig

def generate_static_visualization():
    """Generate complete static HTML with client-side interactivity"""
    
    print("📊 Generating 3D visualization data...")
    data = generate_sample_data()
    
    # Model performance data
    model_performance = {
        'rf': {'auc': 0.85, 'precision': 0.78, 'recall': 0.82, 'name': 'Random Forest'},
        'xgb': {'auc': 0.82, 'precision': 0.75, 'recall': 0.79, 'name': 'XGBoost'},
        'lgb': {'auc': 0.80, 'precision': 0.73, 'recall': 0.77, 'name': 'LightGBM'},
        'svm': {'auc': 0.78, 'precision': 0.70, 'recall': 0.75, 'name': 'SVM'},
        'knn': {'auc': 0.75, 'precision': 0.68, 'recall': 0.72, 'name': 'KNN'}
    }
    
    # Generate initial plot (RF, threshold=0.5)
    fig = create_3d_plot(data, model='rf', threshold=0.5, show_uncertainty=True)
    
    # Pre-compute data for all combinations
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    models = ['rf', 'xgb', 'lgb', 'svm', 'knn']
    
    precomputed_data = {}
    for model in models:
        precomputed_data[model] = {}
        for threshold in thresholds:
            filtered = data[data['probability'] >= threshold]
            precomputed_data[model][str(threshold)] = {
                'points': len(filtered),
                'high_priority': len(filtered[filtered['probability'] >= 0.7]),
                'medium_priority': len(filtered[(filtered['probability'] >= 0.5) & 
                                                (filtered['probability'] < 0.7)]),
                'coverage': len(filtered) / len(data) * 100
            }
    
    # Create HTML with embedded JavaScript for interactivity
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GeoAuPredict 3D Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: {THEME_COLORS['background']};
            color: {THEME_COLORS['foreground']};
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 0.5rem;
            border: 1px solid {THEME_COLORS['border']};
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        label {{
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {THEME_COLORS['accent']};
        }}
        select, input[type="range"] {{
            padding: 0.5rem;
            background: {THEME_COLORS['background']};
            border: 1px solid {THEME_COLORS['border']};
            border-radius: 0.375rem;
            color: {THEME_COLORS['foreground']};
            font-size: 1rem;
        }}
        select {{
            cursor: pointer;
        }}
        select:hover, select:focus {{
            border-color: {THEME_COLORS['accent']};
            outline: none;
        }}
        input[type="range"] {{
            width: 100%;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        .metric-card {{
            background: rgba(30, 41, 59, 0.5);
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid {THEME_COLORS['border']};
        }}
        .metric-card h3 {{
            margin: 0 0 1rem 0;
            color: {THEME_COLORS['accent']};
            font-size: 1.125rem;
        }}
        .metric-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid {THEME_COLORS['border']};
        }}
        .metric-item:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #94a3b8;
        }}
        .metric-value {{
            font-weight: 600;
            color: {THEME_COLORS['foreground']};
        }}
        #plot {{
            background: {THEME_COLORS['background']};
            border-radius: 0.5rem;
            border: 1px solid {THEME_COLORS['border']};
            overflow: hidden;
        }}
        .threshold-value {{
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: {THEME_COLORS['accent']};
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <div class="control-group">
                <label for="model-select">Model</label>
                <select id="model-select" onchange="updateVisualization()">
                    <option value="rf">Random Forest</option>
                    <option value="xgb">XGBoost</option>
                    <option value="lgb">LightGBM</option>
                    <option value="svm">SVM</option>
                    <option value="knn">KNN</option>
                </select>
            </div>
            <div class="control-group">
                <label for="threshold-slider">Probability Threshold</label>
                <input type="range" id="threshold-slider" min="0" max="10" value="5" step="1" oninput="updateVisualization()">
                <div class="threshold-value" id="threshold-value">0.5</div>
            </div>
        </div>
        
        <div id="plot"></div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Model Performance</h3>
                <div id="performance-metrics"></div>
            </div>
            <div class="metric-card">
                <h3>Exploration Targets</h3>
                <div id="target-metrics"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Data embedded from Python
        const rawData = {json.dumps(data.to_dict('list'))};
        const performanceData = {json.dumps(model_performance)};
        const precomputedData = {json.dumps(precomputed_data)};
        
        const themeColors = {{
            background: '{THEME_COLORS['background']}',
            foreground: '{THEME_COLORS['foreground']}',
            accent: '{THEME_COLORS['accent']}',
            border: '{THEME_COLORS['border']}'
        }};
        
        function updateVisualization() {{
            const model = document.getElementById('model-select').value;
            const thresholdIndex = parseInt(document.getElementById('threshold-slider').value);
            const threshold = thresholdIndex / 10;
            
            document.getElementById('threshold-value').textContent = threshold.toFixed(1);
            
            // Filter data by threshold
            const filtered = {{
                lon: [],
                lat: [],
                elev: [],
                prob: [],
                unc: []
            }};
            
            for (let i = 0; i < rawData.probability.length; i++) {{
                if (rawData.probability[i] >= threshold) {{
                    filtered.lon.push(rawData.lon[i]);
                    filtered.lat.push(rawData.lat[i]);
                    filtered.elev.push(rawData.elev[i]);
                    filtered.prob.push(rawData.probability[i]);
                    filtered.unc.push(rawData.uncertainty[i]);
                }}
            }}
            
            // Create hover text
            const hoverText = filtered.prob.map((p, i) => 
                `P: ${{p.toFixed(3)}}<br>U: ${{filtered.unc[i].toFixed(3)}}`
            );
            
            // Update 3D plot
            const trace1 = {{
                type: 'scatter3d',
                mode: 'markers',
                x: filtered.lon,
                y: filtered.lat,
                z: filtered.elev,
                text: hoverText,
                hovertemplate: '<b>Lon:</b> %{{x:.2f}}<br><b>Lat:</b> %{{y:.2f}}<br><b>Elev:</b> %{{z:.0f}}m<br>%{{text}}<extra></extra>',
                marker: {{
                    size: 5,
                    color: filtered.prob,
                    colorscale: [
                        [0, '#262626'],
                        [0.3, '#78350f'],
                        [0.5, '#f59e0b'],
                        [0.7, '#fbbf24'],
                        [1, '#fef3c7']
                    ],
                    cmin: 0,
                    cmax: 1,
                    opacity: 0.8,
                    colorbar: {{
                        title: 'Probability',
                        tickfont: {{ color: themeColors.foreground }},
                        titlefont: {{ color: themeColors.foreground }},
                        bgcolor: 'rgba(0,0,0,0.5)'
                    }}
                }},
                name: 'Probability'
            }};
            
            const trace2 = {{
                type: 'scatter3d',
                mode: 'markers',
                x: filtered.lon,
                y: filtered.lat,
                z: filtered.elev,
                marker: {{
                    size: 3,
                    color: filtered.unc,
                    colorscale: 'Greys',
                    cmin: 0,
                    cmax: 0.5,
                    opacity: 0.3
                }},
                name: 'Uncertainty',
                visible: 'legendonly'
            }};
            
            const layout = {{
                title: {{
                    text: `3D Gold Probability Map - ${{performanceData[model].name}}`,
                    font: {{ color: themeColors.foreground, size: 20 }}
                }},
                scene: {{
                    xaxis: {{
                        title: 'Longitude',
                        backgroundcolor: themeColors.background,
                        gridcolor: themeColors.border,
                        showbackground: true,
                        color: themeColors.foreground
                    }},
                    yaxis: {{
                        title: 'Latitude',
                        backgroundcolor: themeColors.background,
                        gridcolor: themeColors.border,
                        showbackground: true,
                        color: themeColors.foreground
                    }},
                    zaxis: {{
                        title: 'Elevation (m)',
                        backgroundcolor: themeColors.background,
                        gridcolor: themeColors.border,
                        showbackground: true,
                        color: themeColors.foreground
                    }},
                    camera: {{
                        eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                    }}
                }},
                paper_bgcolor: themeColors.background,
                plot_bgcolor: themeColors.background,
                font: {{ color: themeColors.foreground }},
                margin: {{ l: 0, r: 0, t: 40, b: 0 }},
                legend: {{
                    bgcolor: 'rgba(0,0,0,0.5)',
                    font: {{ color: themeColors.foreground }}
                }},
                autosize: true
            }};
            
            const config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            }};
            
            Plotly.newPlot('plot', [trace1, trace2], layout, config);
            
            // Update performance metrics
            const perf = performanceData[model];
            document.getElementById('performance-metrics').innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">AUC Score</span>
                    <span class="metric-value">${{perf.auc.toFixed(3)}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Precision</span>
                    <span class="metric-value">${{perf.precision.toFixed(3)}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Recall</span>
                    <span class="metric-value">${{perf.recall.toFixed(3)}}</span>
                </div>
            `;
            
            // Update target metrics
            const stats = precomputedData[model][threshold.toFixed(1)];
            document.getElementById('target-metrics').innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">High Priority</span>
                    <span class="metric-value">${{stats.high_priority}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Medium Priority</span>
                    <span class="metric-value">${{stats.medium_priority}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Total Targets</span>
                    <span class="metric-value">${{stats.points}}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Coverage</span>
                    <span class="metric-value">${{stats.coverage.toFixed(1)}}%</span>
                </div>
            `;
        }}
        
        // Initial render
        updateVisualization();
        
        // Make responsive
        window.addEventListener('resize', () => {{
            Plotly.Plots.resize('plot');
        }});
    </script>
</body>
</html>
    """
    
    # Save to public directory for Next.js
    output_dir = Path(__file__).parent.parent / 'public'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / '3d-visualization.html'
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated: {output_file}")
    print(f"📊 Data points: {len(data)}")
    print(f"🎯 Models: {len(models)}")
    print(f"📈 Thresholds: {len(thresholds)}")
    print("\n🌐 View at: /3d-visualization.html")
    
    return output_file

if __name__ == '__main__':
    generate_static_visualization()

