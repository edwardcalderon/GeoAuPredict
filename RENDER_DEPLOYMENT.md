# 🚀 Render.com Deployment Guide for GeoAuPredict

## Step-by-Step Instructions

### ✅ Step 1: Push to GitHub (ALREADY DONE!)

Your code is already committed. Just push:

```bash
git push origin main
```

---

### ✅ Step 2: Sign Up on Render.com (2 minutes)

1. Visit: **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with your **GitHub account** (recommended)
4. Authorize Render to access your repositories

---

### ✅ Step 3: Create New Web Service (3 minutes)

1. **After logging in**, click the **"New +"** button (top right)
2. Select **"Web Service"**
3. Click **"Connect a repository"**
4. Find and select **"GeoAuPredict"** from your repo list
   - If you don't see it, click "Configure account" to grant access
5. Click **"Connect"**

---

### ✅ Step 4: Configure Service (Auto-detected!)

Render will automatically detect your `render.yaml` file and show:

```yaml
✓ Name: geoaupredict-api
✓ Environment: Docker
✓ Plan: Free
✓ Health Check: /health
```

**Just click "Create Web Service"** - that's it! 🎉

---

### ✅ Step 5: Wait for Deployment (5-10 minutes)

You'll see the build logs in real-time:

```
==> Cloning from GitHub...
==> Building Docker image...
==> Pushing to registry...
==> Starting service...
==> Health check passed ✓
==> Deploy successful!
```

**Your API URL** will be: `https://geoaupredict-api.onrender.com`

---

## 🧪 Test Your Deployed API

Once deployed, test it:

```bash
# Health check
curl https://geoaupredict-api.onrender.com/health

# Interactive docs
# Visit: https://geoaupredict-api.onrender.com/docs
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "models_available": ["ensemble_gold_v1", "random_forest_model", ...]
}
```

---

## 📊 Test Predictions

```bash
# Make a prediction
curl -X POST "https://geoaupredict-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.5, -0.2, 1.3, 0.1, -0.5, 0.8, 0.3, -1.1, 0.6, 0.2]
    ]
  }'
```

Expected response:
```json
{
  "predictions": [1],
  "probabilities": [0.75],
  "confidence": ["High"]
}
```

---

## ⚠️ Important Notes

### Free Tier Limitations

1. **Cold Starts**: Service sleeps after 15 min of inactivity
   - First request after sleep takes ~30 seconds
   - Subsequent requests are instant
   
2. **512 MB RAM**: Should be enough for the API
   
3. **750 hours/month**: More than enough for 24/7 (744 hours)

### Solutions if Needed

**To eliminate cold starts** (upgrade to $7/month):
1. Go to your service dashboard
2. Settings → Plan
3. Select "Starter" ($7/month)
4. Confirm

---

## 🎯 Monitoring Your Service

### Render Dashboard

Once deployed, you can:

1. **View Logs**: Real-time application logs
2. **Metrics**: CPU, memory, request counts
3. **Environment Variables**: Manage secrets
4. **Manual Deploy**: Redeploy anytime
5. **Custom Domain**: Add your own domain (paid plans)

### Access Dashboard

Visit: https://dashboard.render.com

---

## 🔄 Auto-Deployment

**Good news!** Render automatically redeploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update API"
git push

# Render automatically detects and deploys! 🎉
```

---

## 🐛 Troubleshooting

### Build Fails

**Problem**: Docker build error  
**Solution**: Check build logs in Render dashboard

**Common fixes**:
```bash
# Make sure requirements are up to date
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Health Check Fails

**Problem**: Service unhealthy  
**Solution**: Check that API starts on port 8000

**Test locally first**:
```bash
python src/api/prediction_api.py
# Should show: Uvicorn running on http://0.0.0.0:8000
```

### Models Not Loading

**Problem**: "Model not loaded" error  
**Solution**: Ensure model files are in repo

**Check**:
```bash
ls -lh outputs/models/
# Should show .pkl files
```

---

## 📱 Next Steps After API Deployment

### 1. Deploy Streamlit Dashboards (FREE)

Visit: **https://share.streamlit.io**

Deploy both dashboards:
- `src/app/spatial_validation_dashboard.py`
- `src/app/3d_visualization_dashboard.py`

### 2. Update Website Links

Update your Next.js dashboard page to use the new API URL:

```typescript
const API_URL = 'https://geoaupredict-api.onrender.com';
```

### 3. Add to Documentation

Update your README with:
```markdown
## 🌐 Live Demo

- **REST API**: https://geoaupredict-api.onrender.com
- **API Docs**: https://geoaupredict-api.onrender.com/docs
- **Dashboards**: [Your Streamlit URLs]
```

---

## 💡 Pro Tips

1. **Keep Free Tier Awake**: Use UptimeRobot (free) to ping every 14 minutes
2. **Monitor Performance**: Set up Render's built-in alerts
3. **Version Control**: Tag releases: `git tag v1.0.0 && git push --tags`
4. **Backup Models**: Keep model files in version control
5. **Environment Secrets**: Use Render's environment variables for API keys

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Your Dashboard**: https://dashboard.render.com

---

## ✅ Checklist

- [ ] Push code to GitHub
- [ ] Sign up on Render.com
- [ ] Connect GitHub repository
- [ ] Deploy web service
- [ ] Wait for build to complete
- [ ] Test health endpoint
- [ ] Test predictions endpoint
- [ ] Test interactive docs
- [ ] Add URL to README
- [ ] Deploy Streamlit dashboards (optional)
- [ ] Celebrate! 🎉

---

**Estimated Total Time**: 10-15 minutes  
**Monthly Cost**: $0 (Free tier)  
**Perfect for**: Academic projects, demos, portfolios

Good luck! 🚀

