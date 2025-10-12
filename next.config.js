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
    // CSP headers for production
    async headers() {
        return [
            {
                source: '/(.*)',
                headers: [
                    {
                        key: 'Content-Security-Policy',
                        value: [
                            "default-src 'self'",
                            "script-src 'self' 'unsafe-inline' 'unsafe-hashes' https://polyfill.io https://cdnjs.cloudflare.com",
                            "script-src-attr 'self' 'unsafe-inline' 'unsafe-hashes'",
                            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
                            "font-src 'self' https://fonts.gstatic.com",
                            "img-src 'self' data: https: blob:",
                            "connect-src 'self' http://localhost:8501 http://localhost:8050",
                            "frame-src 'self' http://localhost:8501 http://localhost:8050",
                            "object-src 'none'",
                            "base-uri 'self'",
                            "form-action 'self'"
                        ].join('; ')
                    }
                ]
            }
        ];
    }
};

module.exports = nextConfig;