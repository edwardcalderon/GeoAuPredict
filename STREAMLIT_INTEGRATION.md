# Streamlit Integration with GitHub Pages

## Overview

Your Streamlit dashboard is now deployed at:
**https://gap-geoaupredict.streamlit.app/**

This guide explains how it's integrated with your GitHub Pages site.

## What Was Done

### 1. Updated GitHub Actions Workflow

**File**: `.github/workflows/deploy.yml`

Added the Streamlit URL as an environment variable during build:

```yaml
- name: Build for production
  run: npm run export
  env:
    NODE_ENV: production
    NEXT_PUBLIC_STREAMLIT_URL: https://gap-geoaupredict.streamlit.app
```

### 2. Updated Dashboard Page Logic

**File**: `src/app/dashboards/page.tsx`

**Before**:
- Checked if running on GitHub Pages
- Always showed "run locally" message

**After**:
- Checks if a deployed dashboard URL is available
- If yes → Shows live dashboard in iframe
- If no → Shows helpful local setup instructions

```typescript
// Check if we have a valid deployed dashboard URL (not localhost)
const hasDeployedStreamlit = STREAMLIT_URL && !STREAMLIT_URL.includes('localhost');
const hasDeployedDash = DASH_URL && !DASH_URL.includes('localhost');

// Show helpful message only if we don't have a deployed dashboard
const showStreamlitHelp = isProduction && !hasDeployedStreamlit;
const showDashHelp = isProduction && !hasDeployedDash;
```

## How It Works

### Production (GitHub Pages)
When someone visits https://edwardcalderon.github.io/GeoAuPredict/dashboards/:

1. **Streamlit Tab**: Loads `https://gap-geoaupredict.streamlit.app/?embed=true` in iframe
2. **3D Visualization Tab**: Shows "run locally" (until you deploy Dash)

### Development (Local)
When you run `npm run dev`:

1. Uses `http://localhost:8501` for Streamlit (if running)
2. Uses `http://localhost:8050` for Dash (if running)

## Deployment Flow

```
┌─────────────────────────────────────────┐
│  Push to GitHub (main branch)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  GitHub Actions Triggered               │
│  - Sets NEXT_PUBLIC_STREAMLIT_URL       │
│  - Builds Next.js with env var          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Deployed to GitHub Pages               │
│  - Dashboard page knows Streamlit URL   │
│  - Loads iframe with deployed dashboard │
└─────────────────────────────────────────┘
```

## URLs Summary

| Environment | Streamlit URL | Dash URL |
|-------------|---------------|----------|
| **Production (GitHub Pages)** | `https://gap-geoaupredict.streamlit.app` | Not deployed yet |
| **Local Development** | `http://localhost:8501` | `http://localhost:8050` |

## Next Steps (Optional)

### Deploy Dash 3D Visualization

To also deploy your Dash dashboard:

1. **Option A: Render.com** (Free)
   - Deploy `src/app/dash_3d_app.py` (if exists)
   - Get URL like `https://geoaupredict-dash.onrender.com`

2. **Option B: Railway.app** (Free)
   - Similar process
   - Auto-deploys from GitHub

3. **Update GitHub Actions**:
   ```yaml
   env:
     NODE_ENV: production
     NEXT_PUBLIC_STREAMLIT_URL: https://gap-geoaupredict.streamlit.app
     NEXT_PUBLIC_DASH_URL: https://your-dash-app.onrender.com
   ```

## Testing

### Test Locally
```bash
# Start development server
npm run dev

# Visit http://localhost:3000/dashboards
# Should load dashboards from localhost if running
```

### Test Production Build
```bash
# Build with production settings
NEXT_PUBLIC_STREAMLIT_URL=https://gap-geoaupredict.streamlit.app npm run build

# Check the build
npm start

# Visit http://localhost:3000/dashboards
# Should load from Streamlit Cloud
```

### Test on GitHub Pages
After pushing:
- Visit https://edwardcalderon.github.io/GeoAuPredict/dashboards/
- Should see live Streamlit dashboard

## Troubleshooting

### Dashboard shows "run locally" message on GitHub Pages

**Possible causes**:
1. Environment variable not set in GitHub Actions
2. Build cache needs clearing
3. GitHub Pages not updated yet

**Solution**:
```bash
# Force rebuild
git commit --allow-empty -m "Rebuild for dashboard"
git push
```

### Dashboard loads but has errors

**Check**:
1. Streamlit app is running: https://gap-geoaupredict.streamlit.app/
2. Browser console for CORS errors
3. Streamlit app logs in Streamlit Cloud dashboard

### iframe is blank

**Possible causes**:
1. CORS not configured on Streamlit
2. Content Security Policy blocking iframe
3. Streamlit app crashed

**Solution**:
- Check Streamlit Cloud logs
- Verify CORS settings in `.streamlit/config.toml`

## CORS Configuration

Your Streamlit app needs to allow embedding from GitHub Pages.

**File**: `.streamlit/config.toml` (in Streamlit Cloud)

```toml
[server]
enableCORS = true
enableXsrfProtection = false

[browser]
serverAddress = "gap-geoaupredict.streamlit.app"
```

## Summary

✅ **Streamlit deployed**: https://gap-geoaupredict.streamlit.app/  
✅ **GitHub Actions updated**: Builds with Streamlit URL  
✅ **Dashboard page updated**: Loads live iframe  
✅ **Fallback logic**: Shows help if dashboard not available  

**Result**: Your GitHub Pages site now displays your live Streamlit dashboard! 🎉

