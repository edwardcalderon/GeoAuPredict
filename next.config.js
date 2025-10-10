/** @type {import('next').NextConfig} */

const nextConfig = {
    output: 'export',
    trailingSlash: true,
    images: {
        unoptimized: true,
        domains: ['images.unsplash.com'],
    },
    // GitHub Pages deployment configuration
    basePath: process.env.NODE_ENV === 'production' ? '/geoaupredict' : '',
    assetPrefix: process.env.NODE_ENV === 'production' ? '/geoaupredict/' : '',
};

module.exports = nextConfig;