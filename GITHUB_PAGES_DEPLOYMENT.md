# GitHub Pages Deployment Guide

## ⚠️ Important: GitHub Pages Limitations

GitHub Pages **only hosts static files**. The current dashboard setup with embedded iframes **will NOT work** on GitHub Pages because:

- ❌ Cannot run Python servers (Streamlit, Dash)
- ❌ Cannot run Next.js in server mode
- ❌ Local URLs (`localhost:8501`, `localhost:8050`) won't exist in production

## ✅ Recommended Solution: Hybrid Deployment

Deploy different components to appropriate services:

```
┌─────────────────────────────────────────┐
│  GitHub Pages (Static)                  │
│  - Landing page                         │
│  - Whitepaper                           │
│  - Static content                       │
└─────────────────────────────────────────┘
           │ Links to ↓
┌─────────────────────────────────────────┐
│  Streamlit Cloud (FREE)                 │
│  - Spatial Validation Dashboard         │
│  https://yourapp.streamlit.app          │
└─────────────────────────────────────────┘
           │ Links to ↓
┌─────────────────────────────────────────┐
│  Render/Railway (FREE tier)             │
│  - 3D Visualization Dashboard           │
│  https://yourapp.onrender.com           │
└─────────────────────────────────────────┘
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Deploy Streamlit Dashboard to Streamlit Cloud

**1.1 Prepare Streamlit app:**

Create `streamlit_requirements.txt`:
```bash
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.0.0
folium>=0.14.0
```

**1.2 Push to GitHub:**
```bash
git add src/app/spatial_validation_dashboard.py streamlit_requirements.txt
git commit -m "Add Streamlit dashboard"
git push
```

**1.3 Deploy to Streamlit Cloud:**
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repo: `GeoAuPredict`
5. Set main file: `src/app/spatial_validation_dashboard.py`
6. Set Python version: 3.12
7. Click "Deploy"

**Result:** Your dashboard will be at `https://geoaupredict.streamlit.app`

---

### Step 2: Deploy Dash Dashboard to Render

**2.1 Create `render.yaml`:**

```yaml
services:
  - type: web
    name: geoaupredict-3d
    env: python
    buildCommand: "pip install -r dash_requirements.txt"
    startCommand: "python src/app/3d_visualization_dashboard.py"
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

**2.2 Create `dash_requirements.txt`:**
```bash
dash>=2.14.0
plotly>=5.0.0
pandas>=1.5.0
```

**2.3 Deploy to Render:**
1. Go to [render.com](https://render.com)
2. Sign up/Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your `GeoAuPredict` repo
5. Use existing `render.yaml`
6. Click "Create Web Service"

**Result:** Your 3D dashboard will be at `https://geoaupredict-3d.onrender.com`

---

### Step 3: Update Next.js Dashboard URLs

**3.1 Create environment variables:**

Create `.env.production`:
```bash
NEXT_PUBLIC_STREAMLIT_URL=https://geoaupredict.streamlit.app
NEXT_PUBLIC_DASH_URL=https://geoaupredict-3d.onrender.com
```

**3.2 Update dashboard page:**

```typescript
// src/app/dashboards/page.tsx
const STREAMLIT_URL = process.env.NEXT_PUBLIC_STREAMLIT_URL || 'http://localhost:8501';
const DASH_URL = process.env.NEXT_PUBLIC_DASH_URL || 'http://localhost:8050';

// Then use these variables in iframes:
<iframe src={STREAMLIT_URL} ... />
<iframe src={DASH_URL} ... />
```

---

### Step 4: Deploy Static Next.js to GitHub Pages

**4.1 Update `next.config.js`:**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: process.env.NODE_ENV === 'production' ? '/GeoAuPredict' : '',
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;
```

**4.2 Update `package.json` scripts:**

```json
{
  "scripts": {
    "dev": "next dev",
    "dev:full": "node scripts/dev-with-dashboards.js",
    "build": "next build",
    "export": "next build && touch out/.nojekyll",
    "deploy": "npm run export && gh-pages -d out"
  }
}
```

**4.3 Build and deploy:**

```bash
# Install gh-pages if not already installed
npm install --save-dev gh-pages

# Build and deploy
npm run deploy
```

**Result:** Your static site will be at `https://yourusername.github.io/GeoAuPredict`

---

## 🔧 Configuration Files to Create

### 1. `.env.local` (for development)
```bash
NEXT_PUBLIC_STREAMLIT_URL=http://localhost:8501
NEXT_PUBLIC_DASH_URL=http://localhost:8050
```

### 2. `.env.production` (for production)
```bash
NEXT_PUBLIC_STREAMLIT_URL=https://geoaupredict.streamlit.app
NEXT_PUBLIC_DASH_URL=https://geoaupredict-3d.onrender.com
```

### 3. `streamlit_requirements.txt`
```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.0.0
folium>=0.14.0
geopandas>=0.12.0
```

### 4. `dash_requirements.txt`
```
dash>=2.14.0
plotly>=5.0.0
pandas>=1.5.0
```

---

## 💰 Cost Breakdown

| Service | Plan | Cost | Limitations |
|---------|------|------|-------------|
| **GitHub Pages** | Free | $0 | 100GB bandwidth/month, 1GB storage |
| **Streamlit Cloud** | Community | $0 | 1 app, public only, sleeps after inactivity |
| **Render** | Free Tier | $0 | Sleeps after 15 min inactivity, 750 hours/month |

**Total: FREE** ✨

### Limitations:
- Free tier apps "sleep" after inactivity
- First load may take 30-60 seconds (cold start)
- Limited resources (RAM, CPU)

---

## 🚀 Alternative: All-in-One Deployment

If you want everything in one place:

### Option A: Vercel (Recommended)
- ✅ Host Next.js automatically
- ✅ Can deploy Python dashboards as serverless functions
- ❌ Requires Pro plan for Python backend ($20/month)

### Option B: Railway
- ✅ Host everything together
- ✅ Good free tier
- ❌ More complex setup

### Option C: Heroku
- ✅ Traditional platform
- ❌ No free tier anymore (minimum $5/month per app)

---

## 📝 Updated Workflow

### Development:
```bash
npm run dev:full
# Everything runs locally on localhost
```

### Production:
```
User visits: yourusername.github.io/GeoAuPredict
    ↓
Clicks "Dashboards"
    ↓
Dashboards load from:
  - Streamlit: geoaupredict.streamlit.app
  - Dash: geoaupredict-3d.onrender.com
```

---

## ⚠️ Important Notes

1. **CORS Configuration:** You may need to configure CORS on your Python apps to allow embedding

   **For Streamlit**, add `~/.streamlit/config.toml`:
   ```toml
   [server]
   enableCORS = false
   enableXsrfProtection = false
   ```

2. **Environment Variables:** Never commit production URLs to git. Use `.env.production` (add to `.gitignore`)

3. **SSL/HTTPS:** GitHub Pages, Streamlit Cloud, and Render all provide HTTPS automatically

4. **Custom Domain:** You can use a custom domain with all three services

---

## 🎯 Quick Deploy Commands

```bash
# 1. Build static Next.js site
npm run export

# 2. Deploy to GitHub Pages
npm run deploy

# 3. Streamlit Cloud deploys automatically on git push

# 4. Render deploys automatically on git push
```

---

## 📚 Documentation Links

- [GitHub Pages](https://pages.github.com/)
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Render](https://render.com/docs)
- [Next.js Static Export](https://nextjs.org/docs/app/building-your-application/deploying/static-exports)

---

## ✅ Checklist

- [ ] Deploy Streamlit dashboard to Streamlit Cloud
- [ ] Deploy Dash dashboard to Render/Railway
- [ ] Get production URLs for both dashboards
- [ ] Create `.env.production` with dashboard URLs
- [ ] Update `src/app/dashboards/page.tsx` to use env variables
- [ ] Update `next.config.js` for static export
- [ ] Build and test static export locally
- [ ] Deploy to GitHub Pages
- [ ] Test all links and iframe embeds
- [ ] Set up custom domain (optional)

---

**Ready to deploy? Follow the steps above and you'll have a fully working production deployment!** 🚀

