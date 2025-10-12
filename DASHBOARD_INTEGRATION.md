# GeoAuPredict Dashboard Integration Guide

This guide explains how the Python dashboards (Streamlit and Dash) are integrated with the Next.js web application.

## 🏗️ Architecture

The GeoAuPredict application uses a **hybrid architecture** combining:

- **Next.js Frontend** (TypeScript/React) - Main web interface
- **Streamlit Dashboard** (Python) - Spatial validation visualization
- **Dash Dashboard** (Python) - 3D terrain visualization

```
┌─────────────────────────────────────────────┐
│         Next.js Application                 │
│         http://localhost:3000               │
│  ┌──────────────────────────────────────┐  │
│  │    /dashboards page (iframe embeds)   │  │
│  └──────────────────────────────────────┘  │
└────────┬──────────────────────┬────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Dash 3D       │
│   Port 8501     │    │   Port 8050     │
└─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Automatic Startup (Recommended)

```bash
# Start both dashboards in background
./start_dashboards.sh

# Stop all dashboards
./stop_dashboards.sh
```

### Option 2: Manual Startup

**Terminal 1 - Streamlit:**
```bash
source venv/bin/activate
streamlit run src/app/spatial_validation_dashboard.py
```

**Terminal 2 - Dash:**
```bash
source venv/bin/activate
python src/app/3d_visualization_dashboard.py
```

**Terminal 3 - Next.js:**
```bash
npm run dev
```

## 📍 Access Points

Once all services are running:

- **Main Web App**: http://localhost:3000
- **Dashboards Page**: http://localhost:3000/dashboards
- **Streamlit (Direct)**: http://localhost:8501
- **Dash (Direct)**: http://localhost:8050

## 🎯 Integration Methods

### 1. Iframe Embedding (Current Implementation)

The dashboards are embedded in the Next.js app using iframes:

**File**: `src/app/dashboards/page.tsx`

```tsx
<iframe
  src="http://localhost:8501"
  className="w-full h-full"
  title="Spatial Validation Dashboard"
/>
```

**Pros:**
- ✅ Simple integration
- ✅ Isolates Python/JS environments
- ✅ Maintains full dashboard functionality

**Cons:**
- ❌ Requires both services running
- ❌ Limited cross-frame communication

### 2. API-Based Integration (Future Enhancement)

For production deployments, consider creating API endpoints:

```
Next.js App ↔️ Python API ↔️ Dashboard Data
```

## 📦 Dependencies

### Python Dashboard Dependencies
```bash
# Install dashboard-specific requirements
pip install -r web_requirements.txt
```

Key packages:
- `streamlit>=1.28.0` - Spatial validation UI
- `dash>=2.14.0` - 3D visualization UI
- `plotly>=5.0.0` - Interactive charts
- `folium>=0.14.0` - Map visualizations

### Next.js Dependencies
```bash
npm install
```

## 🔧 Configuration

### Streamlit Configuration

Create `~/.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
```

### Dash Configuration

The Dash app is configured in `src/app/3d_visualization_dashboard.py`:

```python
app.run_server(
    debug=False,
    host='0.0.0.0',
    port=8050
)
```

## 🌐 Production Deployment

### Docker Compose Setup

```yaml
version: '3.8'
services:
  nextjs:
    build: .
    ports:
      - "3000:3000"
    environment:
      - STREAMLIT_URL=http://streamlit:8501
      - DASH_URL=http://dash:8050

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./outputs:/app/outputs

  dash:
    build:
      context: .
      dockerfile: Dockerfile.dash
    ports:
      - "8050:8050"
    volumes:
      - ./outputs:/app/outputs
```

### Nginx Reverse Proxy

For production, use nginx to proxy all services:

```nginx
server {
    listen 80;
    server_name geoaupredict.com;

    # Next.js
    location / {
        proxy_pass http://localhost:3000;
    }

    # Streamlit
    location /streamlit/ {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Dash
    location /dash/ {
        proxy_pass http://localhost:8050/;
    }
}
```

## 🔍 Troubleshooting

### Dashboards Not Loading in Next.js

1. **Check services are running:**
   ```bash
   curl http://localhost:8501
   curl http://localhost:8050
   ```

2. **Check logs:**
   ```bash
   tail -f logs/streamlit.log
   tail -f logs/dash.log
   ```

3. **Verify ports are not blocked:**
   ```bash
   lsof -i :8501
   lsof -i :8050
   ```

### CORS Issues

If you encounter CORS errors, configure your dashboards:

**Streamlit** (`~/.streamlit/config.toml`):
```toml
[server]
enableCORS = false
enableXsrfProtection = false
```

**Dash** (in Python code):
```python
app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
```

### Memory Issues

If dashboards crash due to memory:

```bash
# Increase Node.js memory
export NODE_OPTIONS="--max-old-space-size=4096"

# Monitor memory usage
watch -n 1 'ps aux | grep -E "(streamlit|dash|node)"'
```

## 📊 Dashboard Features

### Streamlit Dashboard
- ✅ Spatial cross-validation results
- ✅ Model performance comparison
- ✅ Interactive probability heat maps
- ✅ Precision@k analysis
- ✅ Geographic confusion matrices

### Dash 3D Dashboard
- ✅ 3D terrain visualization
- ✅ Depth profile analysis
- ✅ Interactive cross-sections
- ✅ Probability volume rendering
- ✅ Export capabilities

## 🔗 Related Files

- `src/app/dashboards/page.tsx` - Next.js integration page
- `src/app/spatial_validation_dashboard.py` - Streamlit app
- `src/app/3d_visualization_dashboard.py` - Dash app
- `web_requirements.txt` - Python dashboard dependencies
- `start_dashboards.sh` - Startup script
- `stop_dashboards.sh` - Shutdown script

## 📝 Development Notes

### Adding New Dashboard Features

1. Edit the respective Python file
2. Restart the dashboard service
3. Refresh the embedded iframe in Next.js

### Modifying the Integration Page

Edit `src/app/dashboards/page.tsx` to customize:
- Tab layout
- Iframe sizes
- Navigation
- Status indicators

## 🚦 Status Monitoring

The dashboards page includes automatic status detection:

```tsx
const [streamlitRunning, setStreamlitRunning] = useState(false);

<iframe
  onLoad={() => setStreamlitRunning(true)}
  onError={() => setStreamlitRunning(false)}
/>
```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Dash Documentation](https://dash.plotly.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Project README](README.md)

## 🤝 Contributing

When adding new visualizations:

1. Create the dashboard in Python (Streamlit/Dash)
2. Add it to the startup scripts
3. Update the Next.js dashboards page
4. Document the new features here

---

**Last Updated**: October 2025  
**Maintainer**: GeoAuPredict Team

