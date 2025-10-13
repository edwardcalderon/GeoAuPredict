# 💰 Cost-Effective Hosting Options for GeoAuPredict

Complete guide to hosting your Docker services affordably.

**Services to Host**:
- REST API (FastAPI)
- Streamlit Dashboards (2)
- MLflow UI
- Static Website (Next.js)

---

## 🏆 Recommended: Hybrid Approach (FREE to $10/month)

### Best Cost-Performance Strategy

| Service | Platform | Cost | Reason |
|---------|----------|------|--------|
| **Static Site** | GitHub Pages | FREE | Already set up |
| **REST API** | Render.com | FREE | Easy Docker deployment |
| **Dashboards** | Streamlit Cloud | FREE | Native Streamlit hosting |
| **MLflow** | Railway.app | $5/mo | Persistent storage needed |

**Total: $5/month (or $0 if you skip MLflow cloud hosting)**

---

## 📊 Detailed Platform Comparison

### 1. **Render.com** ⭐ RECOMMENDED FOR API

#### Pricing
- **Free Tier**: 
  - 750 hours/month (enough for 1 service 24/7)
  - 512 MB RAM
  - Sleeps after 15 min inactivity
  - **COST: $0**

- **Starter**: $7/month
  - Always on
  - 512 MB RAM
  - Better for production

#### Pros
✅ Native Docker support  
✅ Automatic SSL  
✅ GitHub integration (auto-deploy on push)  
✅ Easy environment variables  
✅ Great for REST APIs  

#### Cons
❌ Free tier sleeps (cold starts ~30s)  
❌ Limited to 512MB RAM on free tier  

#### Setup Steps
```bash
# 1. Create render.yaml
cat > render.yaml << 'EOF'
services:
  - type: web
    name: geoaupredict-api
    env: docker
    dockerfilePath: ./Dockerfile
    dockerCommand: python src/api/prediction_api.py
    envVars:
      - key: PORT
        value: 8000
    healthCheckPath: /health
EOF

# 2. Push to GitHub
git add render.yaml
git commit -m "Add Render config"
git push

# 3. Connect on Render.com dashboard
# Visit: https://render.com → New Web Service → Connect Repo
```

**Your API URL**: `https://geoaupredict-api.onrender.com`

---

### 2. **Streamlit Cloud** ⭐ RECOMMENDED FOR DASHBOARDS

#### Pricing
- **Free Tier**:
  - 3 apps
  - 1 GB RAM per app
  - Unlimited viewers
  - **COST: $0**

- **Community Cloud**: FREE FOREVER for public repos

#### Pros
✅ Purpose-built for Streamlit  
✅ Zero configuration  
✅ Direct GitHub integration  
✅ Great performance  
✅ Unlimited bandwidth  

#### Cons
❌ Only for Streamlit apps (not API)  
❌ Must be public repo for free tier  

#### Setup Steps
```bash
# 1. Add requirements.txt for Streamlit apps
cat > streamlit_requirements.txt << 'EOF'
streamlit
pandas
numpy
plotly
geopandas
scikit-learn
EOF

# 2. Deploy
# Visit: https://share.streamlit.io
# Click: "New app" → Select repo → Choose dashboard file
```

**Your Dashboard URLs**:
- `https://share.streamlit.io/edwardcalderon/geoaupredict/spatial`
- `https://share.streamlit.io/edwardcalderon/geoaupredict/viz3d`

---

### 3. **Railway.app** - Good for MLflow

#### Pricing
- **Developer Plan**: $5/month
  - $5 free credits
  - After that: $0.000231/GB-hour
  - ~$3-5/month for light usage

#### Pros
✅ Excellent Docker support  
✅ Persistent volumes  
✅ Great for databases/MLflow  
✅ Easy env vars  
✅ Custom domains  

#### Cons
❌ No free tier anymore  
❌ Slightly more complex than Render  

#### Setup
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

---

### 4. **Fly.io** - Good All-Rounder

#### Pricing
- **Free Tier**:
  - 3 shared-cpu VMs
  - 256MB RAM each
  - 160GB bandwidth/month
  - **COST: $0** for small projects

- **Paid**: ~$1.94/month for 512MB RAM

#### Pros
✅ Generous free tier  
✅ Great Docker support  
✅ Fast global edge deployment  
✅ Persistent volumes  
✅ Can run all services  

#### Cons
❌ Requires credit card even for free tier  
❌ More complex CLI setup  

#### Setup
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Deploy each service
flyctl launch --dockerfile ./Dockerfile
flyctl deploy
```

---

### 5. **DigitalOcean App Platform**

#### Pricing
- **Starter**: $5/month per app
  - 512MB RAM
  - 1 vCPU
  - Good performance

#### Pros
✅ Reliable and fast  
✅ Great for production  
✅ Easy scaling  
✅ Good documentation  

#### Cons
❌ No free tier  
❌ More expensive ($5+ per service)  

---

### 6. **Heroku** (Not Recommended)

#### Status
❌ Removed free tier in November 2022  
❌ Minimum $7/month per dyno  
❌ Better alternatives exist  

---

### 7. **Self-Hosted VPS** - Ultra Budget

#### Providers

| Provider | Plan | RAM | Storage | Cost/Month |
|----------|------|-----|---------|------------|
| **Oracle Cloud** | Always Free | 24GB | 200GB | **$0** |
| **Contabo** | VPS S | 8GB | 200GB | **$3.99** |
| **Hetzner** | CX11 | 2GB | 20GB | **€4.15** |
| **DigitalOcean** | Droplet | 1GB | 25GB | **$6** |
| **Linode** | Nanode | 1GB | 25GB | **$5** |

#### ⭐ **Oracle Cloud Always Free** (BEST VALUE)

**What you get FREE forever**:
- 4 ARM CPU cores (Ampere A1)
- 24 GB RAM
- 200 GB storage
- 10 TB bandwidth/month
- **$0/month forever**

**Setup**:
```bash
# 1. Create Oracle Cloud account
# 2. Launch ARM instance (Ubuntu 22.04)
# 3. SSH and install Docker

ssh ubuntu@your-oracle-ip

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Clone and run
git clone https://github.com/edwardcalderon/GeoAuPredict.git
cd GeoAuPredict
docker compose up -d

# Set up firewall rules in Oracle Cloud dashboard
# Open ports: 8000, 8501, 8502, 5000
```

**Pros**:
✅ FREE FOREVER  
✅ Generous resources (24GB RAM!)  
✅ Run ALL services on one instance  
✅ Full control  
✅ No credit card required for free tier  

**Cons**:
❌ More setup work  
❌ You manage security/updates  
❌ No automatic deployments  

---

## 🏅 Recommended Setup by Budget

### **$0/Month - Maximum Free** 

```
├─ Static Site      → GitHub Pages (FREE)
├─ REST API         → Render.com Free (FREE)
├─ Dashboards (2)   → Streamlit Cloud (FREE)
└─ MLflow           → Skip or run locally
```

**Total: $0/month**

**Limitations**: 
- API sleeps after 15 min (30s cold start)
- No persistent MLflow tracking
- Dashboards must be in public repo

---

### **$5/Month - Best Value** ⭐ RECOMMENDED

```
├─ Static Site      → GitHub Pages (FREE)
├─ REST API         → Render Starter ($7)
├─ Dashboards (2)   → Streamlit Cloud (FREE)
└─ Everything       → Or use Oracle Cloud (FREE)
```

**Option A**: Render + Streamlit = $7/month  
**Option B**: Oracle Cloud VPS = $0/month (but more work)

---

### **$10-15/Month - Production Ready**

```
├─ Static Site      → GitHub Pages (FREE)
├─ REST API         → Render Starter ($7)
├─ Dashboards (2)   → Streamlit Cloud (FREE)
├─ MLflow           → Railway ($5)
└─ OR All-in-one   → Contabo VPS ($4) + backups
```

**Total: $12/month**

**Benefits**:
- No cold starts
- Persistent MLflow
- Good performance
- Scalable

---

### **$20+/Month - Enterprise**

```
All services on DigitalOcean/AWS/GCP
- Better SLA
- Better support
- Auto-scaling
- Load balancing
```

---

## 🎯 My Specific Recommendation for You

### **Best Overall: Hybrid Free Approach**

```yaml
Architecture:
  Static Website:
    Platform: GitHub Pages
    Cost: FREE
    URL: edwardcalderon.github.io/GeoAuPredict
    
  REST API:
    Platform: Render.com (Free)
    Cost: FREE
    URL: geoaupredict-api.onrender.com
    Note: Accepts 30s cold start for demo
    
  Spatial Dashboard:
    Platform: Streamlit Cloud
    Cost: FREE
    URL: Custom subdomain
    
  3D Viz Dashboard:
    Platform: Streamlit Cloud
    Cost: FREE
    URL: Custom subdomain
    
  MLflow:
    Platform: Local only (not hosted)
    Cost: FREE
    Note: Run locally when needed

Total Monthly Cost: $0
```

### **If You Need Always-On API: Add $7/month**

Upgrade Render to Starter plan for:
- No cold starts
- Better reliability
- Custom domain support

---

## 📋 Step-by-Step Deployment Guide

### Phase 1: Deploy Static Site (Already Done)
✅ GitHub Pages is already set up

### Phase 2: Deploy REST API to Render

1. **Create `render.yaml`**:
```bash
cat > render.yaml << 'EOF'
services:
  - type: web
    name: geoaupredict-api
    env: docker
    dockerfilePath: ./Dockerfile
    dockerCommand: python src/api/prediction_api.py
    envVars:
      - key: PORT
        value: 8000
EOF
```

2. **Push to GitHub**:
```bash
git add render.yaml
git commit -m "Add Render deployment config"
git push
```

3. **Deploy on Render**:
   - Visit https://render.com
   - Sign up with GitHub
   - New → Web Service
   - Connect GeoAuPredict repo
   - Render auto-detects render.yaml
   - Deploy!

### Phase 3: Deploy Dashboards to Streamlit Cloud

1. **Visit https://share.streamlit.io**

2. **Deploy Spatial Dashboard**:
   - New app
   - Repo: GeoAuPredict
   - Branch: main
   - Main file: `src/app/spatial_validation_dashboard.py`
   - Deploy!

3. **Deploy 3D Viz Dashboard**:
   - New app
   - Main file: `src/app/3d_visualization_dashboard.py`
   - Deploy!

### Phase 4: Update Links in Your Website

Update dashboard links to point to your deployed services.

---

## 💡 Pro Tips

1. **Use CDN**: Cloudflare (FREE) in front of everything
2. **Monitor**: UptimeRobot (FREE) for uptime monitoring
3. **Analytics**: Plausible or Google Analytics (FREE)
4. **Logs**: Papertrail free tier (100 MB/month)
5. **Secrets**: Never commit API keys, use env vars

---

## 🔒 Security Checklist

- [ ] Use environment variables for secrets
- [ ] Enable HTTPS (automatic on all platforms)
- [ ] Add rate limiting to API
- [ ] Set up CORS properly
- [ ] Regular security updates
- [ ] Monitor for unusual activity

---

## 📊 Cost Summary Table

| Approach | Monthly Cost | Setup Time | Maintenance | Best For |
|----------|--------------|------------|-------------|----------|
| **All Free (Hybrid)** | $0 | 2 hours | Low | Demos, portfolios |
| **Render + Streamlit** | $7 | 1 hour | Very Low | Production |
| **Oracle Cloud** | $0 | 4 hours | Medium | Full control |
| **Fly.io** | $0-6 | 2 hours | Low | Multi-region |
| **Railway** | $5+ | 1 hour | Low | Rapid prototyping |
| **Self-Hosted VPS** | $4-10 | 3 hours | Medium | Budget + control |

---

## 🚀 Quick Start: Deploy in 30 Minutes

```bash
# 1. Deploy API to Render (10 min)
#    → render.com, connect repo, deploy

# 2. Deploy Dashboards to Streamlit (10 min each)
#    → share.streamlit.io, connect repo, deploy

# 3. Update website links (5 min)
#    → Update URLs in dashboards/page.tsx

# 4. Test everything (5 min)
#    → Visit all URLs, verify functionality

Total time: ~35 minutes
Total cost: $0/month
```

---

## 📞 Need Help?

- **Render Docs**: https://render.com/docs
- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Oracle Cloud**: https://docs.oracle.com/cloud/
- **Fly.io Docs**: https://fly.io/docs/

---

**My Recommendation**: Start with the **All Free (Hybrid)** approach. It's perfect for academic projects, demos, and portfolios. You can always upgrade to paid plans later if needed.

**Total Setup Time**: ~2 hours  
**Monthly Cost**: $0  
**Perfect For**: Final project presentations! 🎓

