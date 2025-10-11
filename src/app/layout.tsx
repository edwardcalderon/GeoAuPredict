import { TempoInit } from "@/components/tempo-init";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "GeoAuPredict G.A.P - AI-Driven Geospatial Gold Prediction",
  description: "Advanced AI-powered system that integrates multiple geospatial data sources to predict gold presence in subsoil with interactive 3D visualization and confidence levels for mining exploration.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* CSP-compliant script loading for production */}
        {process.env.NODE_ENV === 'production' ? (
          <>
            <script
              dangerouslySetInnerHTML={{
                __html: `
                  (function() {
                    var script = document.createElement('script');
                    script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
                    script.async = true;
                    document.head.appendChild(script);
                  })();
                `
              }}
            />
            <script
              dangerouslySetInnerHTML={{
                __html: `
                  (function() {
                    var script = document.createElement('script');
                    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML';
                    script.async = true;
                    document.head.appendChild(script);
                  })();
                `
              }}
            />
          </>
        ) : (
          <>
            {/* MathJax for LaTeX rendering - Development */}
            <Script
              src="https://polyfill.io/v3/polyfill.min.js?features=es6"
              strategy="beforeInteractive"
            />
            <Script
              src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
              strategy="beforeInteractive"
            />
          </>
        )}
      </head>
      <body className={inter.className}>
        {children}
        <TempoInit />
      </body>
    </html>
  );
}
