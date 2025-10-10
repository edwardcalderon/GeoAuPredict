# Deployment to GitHub Pages

This project is configured for automatic deployment to GitHub Pages using GitHub Actions.

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository settings on GitHub
2. Navigate to "Pages" section
3. Under "Source", select "GitHub Actions"

### 2. Repository Settings

The deployment is configured to:
- Build on every push to the `main` branch
- Deploy the static export to GitHub Pages
- Use the repository name as the base path (`/geoaupredict/`)

## Configuration

### Next.js Configuration (`next.config.js`)

```javascript
    output: 'export',
    trailingSlash: true,
    images: {
        unoptimized: true,
    },
    basePath: process.env.NODE_ENV === 'production' ? '/geoaupredict' : '',
    assetPrefix: process.env.NODE_ENV === 'production' ? '/geoaupredict/' : '',
};

### GitHub Actions Workflow
- Node.js setup
- Dependencies installation
- Static export generation
- Deployment to GitHub Pages

## Deployment Commands

### Manual Deployment

```bash
# Install dependencies
npm install

# Build for production
npm run export

# Add .nojekyll file for GitHub Pages
echo "" > out/.nojekyll

# Deploy to GitHub Pages
npm run deploy
```

### Development

```bash
# Start development server
npm run dev

# Build for production (without deployment)
npm run build
```

## Important Notes

1. **Static Export**: The app is configured for static export, which means:
   - No server-side rendering
   - No API routes
   - All routes must be known at build time

2. **GitHub Pages Limitations**:
   - Only supports static files
   - No server-side processing
   - Custom 404 page must be handled in the app

3. **Images**: Images are unoptimized for static export compatibility

4. **Base Path**: The app uses `/geoaupredict` as the base path. Update this in `next.config.js` if your base path is different.

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Check that all dependencies are installed: `npm install`
2. Ensure Node.js version 18+ is being used
3. Check for any dynamic routes that aren't properly handled

### Deployment Issues

If deployment fails:

1. Check the GitHub Actions logs in your repository
2. Ensure GitHub Pages is enabled in repository settings
3. Verify that the `main` branch exists and is up to date

### Custom Domain

To use a custom domain:

2. Add a `CNAME` file to your repository with your domain name
3. Configure your domain's DNS settings to point to GitHub Pages

## Live URL

After successful deployment, your site will be available at:
`https://yourusername.github.io/geoaupredict/`

Replace `yourusername` with your actual GitHub username.
