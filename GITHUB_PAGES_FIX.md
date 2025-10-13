# GitHub Pages Dashboard Fix

## Problem

When visiting `https://edwardcalderon.github.io/GeoAuPredict/dashboards/`, the page showed:
- URL: `dashboards/undefined?embed=true`
- Empty iframe with no dashboard content

## Root Cause

1. **GitHub Pages = Static Site Only**
   - Cannot run Python backends (Streamlit/Dash)
   - Only serves HTML, CSS, JavaScript files

2. **Undefined Environment Variables**
   - `NEXT_PUBLIC_STREAMLIT_URL` was not set
   - `NEXT_PUBLIC_DASH_URL` was not set
   - Code tried to load from `undefined` URL

## Solution Implemented

### 1. Fixed Dashboard Page (`src/app/dashboards/page.tsx`)

**Before:**
```typescript
const STREAMLIT_URL = isProduction ? process.env.NEXT_PUBLIC_STREAMLIT_URL : 'http://localhost:8501';
// Result on GitHub Pages: undefined
```

**After:**
```typescript
const STREAMLIT_URL = process.env.NEXT_PUBLIC_STREAMLIT_URL || 'http://localhost:8501';
const isGitHubPages = isProduction && window.location.hostname.includes('github.io');
```

### 2. Added Helpful UI for GitHub Pages Users

Now shows a clear message explaining:
- ✅ Why dashboards aren't available on GitHub Pages
- ✅ How to run dashboards locally
- ✅ Installation instructions
- ✅ Nice visual design

### 3. Created Documentation

- **`DASHBOARD_DEPLOYMENT.md`**: Complete guide for deploying dashboards separately
- **`STREAMLIT_SETUP.md`**: Streamlit Cloud deployment configuration

## Current Behavior

### On GitHub Pages (Production)
- Shows informative message
- Provides local setup instructions
- Clear, professional UI

### On Localhost (Development)
- Loads dashboards from `http://localhost:8501` and `http://localhost:8050`
- Full functionality when dashboards are running

### With Deployed Backends (Future)
- Set environment variables:
  - `NEXT_PUBLIC_STREAMLIT_URL=https://your-app.streamlit.app`
  - `NEXT_PUBLIC_DASH_URL=https://your-dash-app.onrender.com`
- Dashboards will load in iframes

## Files Modified

1. ✅ `src/app/dashboards/page.tsx` - Fixed undefined URLs, added helpful UI
2. ✅ `DASHBOARD_DEPLOYMENT.md` - Deployment guide
3. ✅ `STREAMLIT_SETUP.md` - Configuration guide
4. ✅ `requirements.txt` - Minimal for deployment
5. ✅ `requirements_full.txt` - Complete for development
6. ✅ `environment_dev.yml` - Renamed to avoid conda on Streamlit

## Testing

1. **Local Development:**
   ```bash
   npm run dev
   # Visit http://localhost:3000/dashboards
   # Should show local dashboard iframes (if running)
   ```

2. **Production Build:**
   ```bash
   npm run build
   npm start
   # Visit http://localhost:3000/dashboards
   # Should show helpful message
   ```

3. **GitHub Pages:**
   - Visit `https://edwardcalderon.github.io/GeoAuPredict/dashboards/`
   - Should see helpful message with setup instructions

## Next Steps (Optional)

To enable live dashboards on your GitHub Pages site:

### Quick Option (Free):
1. Deploy Streamlit dashboard to Streamlit Community Cloud
2. Add environment variable to GitHub Actions:
   ```yaml
   NEXT_PUBLIC_STREAMLIT_URL: https://your-app.streamlit.app
   ```
3. Rebuild and redeploy GitHub Pages

### Full Option (Free or Paid):
1. Deploy both dashboards to Render.com (free tier)
2. Configure CORS settings
3. Update environment variables
4. Rebuild GitHub Pages

See `DASHBOARD_DEPLOYMENT.md` for complete instructions.

## Summary

✅ **Fixed**: Undefined dashboard URLs  
✅ **Improved**: User experience with clear messaging  
✅ **Documented**: Multiple deployment options  
✅ **Ready**: For separate dashboard deployment  

The site now gracefully handles the limitation of GitHub Pages and provides a professional user experience!

