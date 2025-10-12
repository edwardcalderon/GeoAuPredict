# GeoAuPredict Quick Start Guide

## 🚀 Automatic Startup (Recommended)

Start everything with **one command**:

```bash
npm run dev:full
```

This automatically starts:
- ✅ Streamlit Dashboard (port 8501)
- ✅ Dash 3D Dashboard (port 8050)  
- ✅ Next.js Web App (port 3000)

Then open your browser to:
**http://localhost:3000/dashboards**

### Stop All Services

Press `Ctrl+C` in the terminal to stop everything at once.

---

## 📋 Alternative Startup Methods

### Method 1: Dashboards Only

Start just the Python dashboards:

```bash
npm run dev:dashboards
# or
./start_dashboards.sh
```

Then in another terminal:

```bash
npm run dev
```

Stop dashboards:

```bash
npm run stop:dashboards
# or
./stop_dashboards.sh
```

### Method 2: Manual Startup (3 Terminals)

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

---

## 🎯 Access Points

Once running, access:

| Service | URL | Description |
|---------|-----|-------------|
| **Main App** | http://localhost:3000 | Next.js landing page |
| **Dashboards Hub** | http://localhost:3000/dashboards | Integrated dashboard view |
| **Streamlit** | http://localhost:8501 | Spatial validation (direct) |
| **Dash 3D** | http://localhost:8050 | 3D visualization (direct) |

---

## 🔧 First Time Setup

### 1. Install Dependencies

```bash
# Python dependencies
source venv/bin/activate
pip install -r requirements.txt
pip install -r web_requirements.txt

# Node.js dependencies
npm install
```

### 2. Run Spatial Validation Notebook

```bash
jupyter notebook notebooks/GeoAuPredict_Spatial_Validation.ipynb
```

Run all cells to generate:
- `outputs/spatial_validation_probability_map.tif`
- `outputs/spatial_validation_heatmap.html`
- `outputs/spatial_validation_results.json`

### 3. Start the Application

```bash
npm run dev:full
```

---

## 📊 Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Next.js only |
| `npm run dev:full` | **Start everything automatically** |
| `npm run dev:dashboards` | Start Python dashboards only |
| `npm run stop:dashboards` | Stop Python dashboards |
| `npm run build` | Build Next.js for production |
| `npm run start` | Start production build |

---

## 🐛 Troubleshooting

### Services Won't Start

**Check if ports are already in use:**
```bash
lsof -i :3000  # Next.js
lsof -i :8501  # Streamlit
lsof -i :8050  # Dash
```

**Kill existing processes:**
```bash
npm run stop:dashboards
pkill -f streamlit
pkill -f "python.*3d_visualization"
```

### Python Dependencies Missing

```bash
source venv/bin/activate
pip install -r web_requirements.txt
```

### Data Files Not Found

Make sure you've run the spatial validation notebook first to generate outputs.

### Port Already in Use

Edit the ports in:
- `scripts/dev-with-dashboards.js` (Streamlit/Dash)
- `next.config.js` (Next.js)

---

## 📁 Project Structure

```
GeoAuPredict/
├── src/app/
│   ├── page.tsx                              # Landing page
│   ├── dashboards/page.tsx                   # Dashboard hub
│   ├── spatial_validation_dashboard.py       # Streamlit app
│   └── 3d_visualization_dashboard.py         # Dash app
├── scripts/
│   └── dev-with-dashboards.js                # Auto-startup script
├── outputs/                                  # Generated results
├── notebooks/                                # Jupyter notebooks
└── package.json                              # NPM scripts
```

---

## 🎓 Next Steps

1. **Explore the Dashboards**: http://localhost:3000/dashboards
2. **Read the White Paper**: http://localhost:3000/whitepaper
3. **Run Custom Analyses**: Modify notebooks and dashboards
4. **Deploy**: See `DEPLOYMENT.md` for production setup

---

## 📚 More Documentation

- **Dashboard Integration**: `DASHBOARD_INTEGRATION.md`
- **Full Setup**: `README.md`
- **Deployment Guide**: `DEPLOYMENT.md`
- **Notebook Guide**: `notebooks/NOTEBOOK_GUIDE.md`

---

## 💡 Pro Tips

- Use `npm run dev:full` for the best development experience
- Check `logs/` directory if services fail to start
- The dashboards page shows service status indicators
- All services stop together with `Ctrl+C`

---

**Happy Exploring! 🌟**

