# 🔐 Google OAuth Setup Guide

## Issue Fixed
The redirect URL was missing the GitHub Pages basePath (`/GeoAuPredict`). This has been fixed in the code.

## Required Configuration

### 1. Supabase Dashboard Configuration

Go to: https://app.supabase.com/project/gfncvtssohgbqgmcjblp/auth/url-configuration

#### Site URL
```
https://edwardcalderon.github.io/GeoAuPredict
```

#### Redirect URLs (Add ALL of these)
```
https://edwardcalderon.github.io/GeoAuPredict/dashboards
https://edwardcalderon.github.io/GeoAuPredict/dashboards/
https://edwardcalderon.github.io/GeoAuPredict/**
http://localhost:3000/dashboards
http://localhost:3000/dashboards/
```

### 2. Google Cloud Console Configuration

Go to: https://console.cloud.google.com/apis/credentials

Find your OAuth 2.0 Client ID and add these **Authorized redirect URIs**:

```
https://gfncvtssohgbqgmcjblp.supabase.co/auth/v1/callback
```

**Authorized JavaScript origins:**
```
https://edwardcalderon.github.io
https://gfncvtssohgbqgmcjblp.supabase.co
http://localhost:3000
```

### 3. Enable Google Provider in Supabase

Go to: https://app.supabase.com/project/gfncvtssohgbqgmcjblp/auth/providers

1. ✅ **Enable Google Provider**
2. Enter your **Client ID** from Google Cloud Console
3. Enter your **Client Secret** from Google Cloud Console
4. Save changes

## Testing

### Local Development
1. Navigate to: `http://localhost:3000/login`
2. Click "Continue with Google"
3. Should redirect to: `http://localhost:3000/dashboards`

### Production
1. Navigate to: `https://edwardcalderon.github.io/GeoAuPredict/login`
2. Click "Continue with Google"
3. Should redirect to: `https://edwardcalderon.github.io/GeoAuPredict/dashboards`

## Debugging

Open the browser console and look for:
```
🔐 OAuth Redirect URL: https://edwardcalderon.github.io/GeoAuPredict/dashboards
```

If you see `http://localhost:3000/` in production, the environment variable `NODE_ENV` might not be set correctly during the build.

## After Making Changes

1. Commit the code changes:
   ```bash
   git add src/lib/supabase.ts src/app/login/page.tsx
   git commit -m "fix: include basePath in OAuth redirect URL for GitHub Pages"
   ```

2. Push to trigger deployment:
   ```bash
   git push origin main
   ```

3. Wait for GitHub Actions to deploy (~2-5 minutes)

4. Test the login flow on production

## Common Issues

❌ **Still redirecting to localhost?**
- Clear browser cache and cookies
- Check that the GitHub Actions build succeeded
- Verify Supabase redirect URLs are saved correctly

❌ **"Redirect URL not allowed" error?**
- Double-check the Supabase Redirect URLs list
- Make sure to click "Save" in Supabase Dashboard
- URLs are case-sensitive!

❌ **OAuth popup closes immediately?**
- Check Google Cloud Console redirect URIs
- Ensure the Supabase callback URL is added

## Support

If issues persist, check:
- GitHub Actions logs: https://github.com/edwardcalderon/GeoAuPredict/actions
- Supabase logs: https://app.supabase.com/project/gfncvtssohgbqgmcjblp/logs/auth-logs
- Browser console for error messages

