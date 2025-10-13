import { TempoInit } from "@/components/tempo-init";
import { AuthProvider } from "@/contexts/AuthContext";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "GAP ( GeoAuPredict) | AI-Driven Geospatial Gold Prediction",
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
        {/* CSP-compliant: No external scripts for GitHub Pages compatibility */}
        {/* MathJax removed due to GitHub Pages CSP restrictions */}
        {/* If LaTeX rendering is needed, consider alternative approaches */}
      </head>
      <body className={inter.className}>
        <AuthProvider>
          {children}
        </AuthProvider>
        <TempoInit />
      </body>
    </html>
  );
}
