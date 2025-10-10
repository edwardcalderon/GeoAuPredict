/** @type {import('next').NextConfig} */

const nextConfig = {
    output: 'export',
    trailingSlash: true,
    images: {
        unoptimized: true,
        domains: ['images.unsplash.com'],
    },
    // GitHub Pages deployment configuration
    basePath: process.env.NODE_ENV === 'production' ? '/GeoAuPredict' : '',
    assetPrefix: process.env.NODE_ENV === 'production' ? '/GeoAuPredict/' : '',
};

module.exports = nextConfig;