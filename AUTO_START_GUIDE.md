# 🚀 Automatic Dashboard Startup - Complete Guide

## ✨ What's New?

The dashboards now **start automatically** with Next.js! No more juggling multiple terminals.

## 🎯 One Command to Rule Them All

```bash
npm run dev:full
```

That's it! 🎉

## 🔍 What Happens?

When you run `npm run dev:full`:

```
Starting services...
┌─────────────────────┐
│  Checking Python    │
│  dependencies...    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Starting Streamlit │
│  (port 8501)...     │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Starting Dash 3D   │
│  (port 8050)...     │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Starting Next.js   │
│  (port 3000)...     │
└─────────────────────┘
         ↓
✅ All services running!
```

## 📊 Output You'll See

```
============================================================
🌟  GeoAuPredict Development Environment
============================================================

📦 Checking Python dependencies...
✅ Python dependencies ready

📍 Starting services...

🚀 Starting Streamlit...
✅ Streamlit started
🚀 Starting Dash...
✅ Dash started
🚀 Starting Next.js...
✅ Next.js started

============================================================
✅  All services running!
============================================================

📊 Access your application:
   • Main App:      http://localhost:3000
   • Dashboards:    http://localhost:3000/dashboards
   • Streamlit:     http://localhost:8501
   • Dash 3D:       http://localhost:8050

📋 Logs:
   • Streamlit:     logs/streamlit.log
   • Dash:          logs/dash.log

🛑 Press Ctrl+C to stop all services
```

## 🛑 Stopping Services

Just press `Ctrl+C` in the terminal where services are running.

All three services will stop automatically:
```
🛑 Stopping all services...
   Stopping Streamlit...
   Stopping Dash...
   Stopping Next.js...
✅ All dashboard processes stopped
```

## 📝 Available Commands

| Command | What It Does | When to Use |
|---------|--------------|-------------|
| **`npm run dev:full`** | **Starts everything automatically** | **🌟 Use this for development!** |
| `npm run dev` | Start Next.js only | When you just need the web app |
| `npm run dev:dashboards` | Start Python dashboards only | When Next.js is already running |
| `npm run stop:dashboards` | Stop Python dashboards | Clean up dashboard processes |

## 🎨 Features

### ✅ Automatic Management
- All processes start and stop together
- No orphaned processes
- Clean shutdown with Ctrl+C

### ✅ Smart Logging
- Logs written to `logs/` directory
- Color-coded console output
- Filter out common warnings

### ✅ Dependency Check
- Automatically checks Python packages
- Installs missing dependencies
- Ensures everything is ready

### ✅ Health Monitoring
- Detects if services fail to start
- Shows startup confirmation
- Reports errors clearly

## 🔧 How It Works

The `scripts/dev-with-dashboards.js` script:

1. **Checks Environment**
   - Verifies venv exists
   - Installs Python dependencies
   - Creates logs directory

2. **Starts Services Sequentially**
   - Streamlit first (needs more time)
   - Dash second
   - Next.js last (fastest)

3. **Manages Processes**
   - Captures all output
   - Handles errors gracefully
   - Cleanup on exit

4. **Stops Everything Together**
   - SIGTERM to all processes
   - Force kill after 5 seconds
   - No zombie processes

## 🐛 Troubleshooting

### Services Don't Start

**Check if ports are in use:**
```bash
lsof -i :3000,8501,8050
```

**Kill existing processes:**
```bash
npm run stop:dashboards
```

### Python Dependencies Error

```bash
source venv/bin/activate
pip install -r web_requirements.txt
```

### Can't Find venv

Make sure you're in the project root:
```bash
cd /home/ed/Documents/maestria/GeoAuPredict
```

Create venv if missing:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Services Start but Won't Stop

Force kill all:
```bash
pkill -f streamlit
pkill -f "python.*3d_visualization"
pkill -f "next-server"
```

## 📁 Generated Files

When running, these files are created/used:

```
GeoAuPredict/
├── logs/
│   ├── streamlit.log        # Streamlit console output
│   └── dash.log              # Dash console output
├── .dashboard_pids           # Process IDs (for cleanup)
└── scripts/
    └── dev-with-dashboards.js
```

## 🔄 Comparison: Before vs After

### Before (3 terminals needed):

**Terminal 1:**
```bash
source venv/bin/activate
streamlit run src/app/spatial_validation_dashboard.py
```

**Terminal 2:**
```bash
source venv/bin/activate
python src/app/3d_visualization_dashboard.py
```

**Terminal 3:**
```bash
npm run dev
```

### After (1 terminal):

```bash
npm run dev:full
```

**That's it!** 🎉

## 🎯 Best Practices

### For Development
```bash
npm run dev:full
```
- Use this as your default
- Everything starts together
- Single Ctrl+C stops all

### For Testing Next.js Only
```bash
npm run dev
```
- When you don't need dashboards
- Faster startup
- Less resource usage

### For Production
See `DEPLOYMENT.md` for Docker Compose setup with proper service orchestration.

## 🚦 Service Status

You can check service status from the Next.js dashboard page:

http://localhost:3000/dashboards

The page shows:
- 🟢 Green badge = Service running
- ⚫ Gray badge = Service stopped

## 💡 Pro Tips

1. **Keep one terminal dedicated**
   - Let the services run in one terminal
   - Do other work in other terminals

2. **Check logs if issues**
   ```bash
   tail -f logs/streamlit.log
   tail -f logs/dash.log
   ```

3. **Use status indicators**
   - Visit /dashboards page
   - Check badges for service status

4. **Clean restart if needed**
   ```bash
   npm run stop:dashboards
   npm run dev:full
   ```

## 🎓 Next Steps

1. Run `npm run dev:full`
2. Open http://localhost:3000/dashboards
3. Explore the embedded dashboards
4. Check out the 3D visualizations
5. Review spatial validation results

## 📚 Related Documentation

- **Quick Start**: `QUICK_START.md` - Basic usage guide
- **Integration**: `DASHBOARD_INTEGRATION.md` - Technical details
- **Summary**: `INTEGRATION_SUMMARY.md` - Overview of changes
- **Main README**: `README.md` - Full project documentation

---

## ✨ Summary

| Before | After |
|--------|-------|
| 3 terminals | 1 terminal |
| Manual startup | Automatic startup |
| Manual cleanup | Automatic cleanup |
| Complex | **Simple** ✨ |

---

**🎉 Enjoy your streamlined development workflow!**

