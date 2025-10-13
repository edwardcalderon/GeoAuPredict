# Dashboard Deployment Guide

## Overview

GeoAuPredict includes two interactive dashboards:
1. **Streamlit Dashboard** - Spatial validation and analysis (Port 8501)
2. **Dash Dashboard** - 3D visualization (Port 8050)

## Why Dashboards Don't Work on GitHub Pages

GitHub Pages only serves **static files** (HTML, CSS, JavaScript). The dashboards require:
- ❌ Python runtime
- ❌ Backend server processes
- ❌ Dynamic data processing

**Solution**: Deploy dashboards separately on a platform that supports Python backends.

---

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for Streamlit)

**Free tier available** ✨

1. **Prepare Your Repo**
   ```bash
   # Ensure you have:
   # - requirements.txt (already created)
   # - src/app/spatial_validation_dashboard.py
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `edwardcalderon/GeoAuPredict`
   - Main file path: `src/app/spatial_validation_dashboard.py`
   - Click "Deploy"

3. **Get Your URL**
   - After deployment, you'll get a URL like:
   - `https://your-app-name.streamlit.app`

4. **Update Your Next.js App**
   - Add to your GitHub repository secrets or environment variables:
   ```
   NEXT_PUBLIC_STREAMLIT_URL=https://your-app-name.streamlit.app
   ```

### Option 2: Render.com (Good for Both Dashboards)

**Free tier available** ✨

#### For Streamlit:
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - **Name**: `geoaupredict-streamlit`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run src/app/spatial_validation_dashboard.py --server.port=$PORT`
   - **Instance Type**: Free
5. Add environment variable:
   - `STREAMLIT_SERVER_HEADLESS=true`

#### For Dash:
1. Similar process
2. **Start Command**: `python src/app/dash_3d_app.py`

### Option 3: Heroku

**Requires credit card** (but has free tier)

1. Create `Procfile`:
   ```
   web: streamlit run src/app/spatial_validation_dashboard.py --server.port=$PORT
   ```

2. Deploy:
   ```bash
   heroku create geoaupredict-dashboard
   heroku git:remote -a geoaupredict-dashboard
   git push heroku main
   ```

### Option 4: Railway.app

**Free tier available** ✨

1. Go to [railway.app](https://railway.app)
2. "Start a New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Railway auto-detects Python and uses `requirements.txt`
5. Add start command in settings:
   ```
   streamlit run src/app/spatial_validation_dashboard.py
   ```

### Option 5: Local Development Only

Keep dashboards for local use only:
```bash
# Clone and setup
git clone https://github.com/edwardcalderon/GeoAuPredict.git
cd GeoAuPredict
pip install -r requirements_full.txt

# Start dashboards
./start_dashboards.sh

# Access:
# Streamlit: http://localhost:8501
# Dash: http://localhost:8050
```

---

## Connecting Deployed Dashboards to GitHub Pages

Once you deploy your dashboards, update your Next.js site:

### Method 1: Environment Variables (During Build)

Update your GitHub Actions workflow (`.github/workflows/deploy.yml`):

```yaml
- name: Build Next.js
  env:
    NEXT_PUBLIC_STREAMLIT_URL: https://your-app.streamlit.app
    NEXT_PUBLIC_DASH_URL: https://your-dash-app.onrender.com
  run: npm run build
```

### Method 2: Hardcode in Config

Update `src/app/dashboards/page.tsx`:

```typescript
const STREAMLIT_URL = process.env.NEXT_PUBLIC_STREAMLIT_URL || 
                      'https://your-app.streamlit.app';
const DASH_URL = process.env.NEXT_PUBLIC_DASH_URL || 
                 'https://your-dash-app.onrender.com';
```

---

## Security Considerations

### CORS Configuration

Your deployed dashboards need to allow requests from GitHub Pages.

**For Streamlit** (`.streamlit/config.toml`):
```toml
[server]
enableCORS = true
enableXsrfProtection = false

[browser]
serverAddress = "your-domain.streamlit.app"
```

**For Dash**:
```python
from flask_cors import CORS

# In your Dash app initialization
CORS(app.server, origins=["https://edwardcalderon.github.io"])
```

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Streamlit Cloud | ✅ 1 app | $20/mo | Streamlit only |
| Render.com | ✅ 750 hrs/mo | $7/mo | Both dashboards |
| Railway.app | ✅ $5 credit/mo | Pay as you go | Small projects |
| Heroku | ⚠️ CC required | $7/mo | Production apps |
| Local Only | ✅ Free | - | Development |

---

## Recommended Setup

**For MVP/Demo**:
- Deploy Streamlit on **Streamlit Community Cloud** (free)
- Keep Dash for local use
- Update GitHub Pages with Streamlit URL

**For Production**:
- Deploy both on **Render.com** ($7/mo total)
- Custom domain support
- Better uptime and performance

---

## Current Status

✅ **GitHub Pages**: Live at `https://edwardcalderon.github.io/GeoAuPredict`  
⚠️ **Dashboards**: Currently show instructions to run locally  
📝 **Next Step**: Deploy dashboards to Streamlit Cloud/Render

---

## Troubleshooting

### Dashboard iframe not loading
- Check CORS configuration
- Verify dashboard URL is accessible
- Check browser console for errors

### Environment variables not working
- Ensure variables are prefixed with `NEXT_PUBLIC_`
- Rebuild Next.js after adding environment variables
- Check if variables are set in build environment

### Dashboard shows "undefined"
- Environment variables not set properly
- Fallback to local URLs
- Code now shows helpful message on GitHub Pages

