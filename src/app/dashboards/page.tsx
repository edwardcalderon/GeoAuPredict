'use client';

import AppHeader from '@/components/AppHeader';
import Footer from '@/components/Footer';
import ProtectedRoute from '@/components/ProtectedRoute';
import Viz3D from '@/components/Viz3D';
import NotebookViewer from '@/components/NotebookViewer';
import { Map, Box, BookOpen } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Dashboard URLs - use environment variables for production deployment
const isProduction = process.env.NODE_ENV === 'production';

// In development, ALWAYS use localhost. In production, use env vars or show help.
const STREAMLIT_URL = isProduction 
  ? (process.env.NEXT_PUBLIC_STREAMLIT_URL || '')
  : 'http://localhost:8501';

// Check if we have a valid deployed dashboard URL (not localhost)
const hasDeployedStreamlit = isProduction && STREAMLIT_URL && !STREAMLIT_URL.includes('localhost');

// Show helpful message only if in production without a deployed dashboard
const showStreamlitHelp = isProduction && !hasDeployedStreamlit;


export default function DashboardsPage() {
  return (
    <ProtectedRoute>
      <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col overflow-hidden">
        {/* Fixed Header */}
        <div className="flex-shrink-0">
          <AppHeader currentPage="dashboards" />
        </div>

        {/* Main Content - Scrollable */}
        <main className="flex-1 overflow-hidden flex flex-col">
          <div className="flex-1 flex flex-col px-4 py-4 overflow-hidden max-w-7xl mx-auto w-full">
            {/* Page Header */}
            <div className="flex-shrink-0 mb-4 flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Gold Prediction Analysis
                </h1>
                <p className="text-sm text-slate-400">
                  Interactive dashboards, 3D visualization, and project notebook
                </p>
              </div>
            </div>

            {/* Tabs Component - Takes remaining space */}
            <Tabs defaultValue="notebook" className="flex-1 flex flex-col overflow-hidden">
              <TabsList className="flex-shrink-0 grid w-full max-w-2xl mx-auto grid-cols-3 bg-slate-800/50 border border-slate-700 rounded-lg mb-4" style={{ pointerEvents: 'auto' }}>
              <TabsTrigger
                value="notebook"
                className="flex items-center justify-center data-[state=active]:bg-slate-700 data-[state=active]:text-yellow-400 text-slate-400 cursor-pointer"
                style={{ pointerEvents: 'auto' }}
              >
                <BookOpen className="w-4 h-4 mr-2" />
                Notebook
              </TabsTrigger>
              <TabsTrigger
                value="spatial"
                className="flex items-center justify-center data-[state=active]:bg-slate-700 data-[state=active]:text-yellow-400 text-slate-400 cursor-pointer"
                style={{ pointerEvents: 'auto' }}
              >
                <Map className="w-4 h-4 mr-2" />
                Spatial Validation
              </TabsTrigger>
              <TabsTrigger
                value="3d"
                className="flex items-center justify-center data-[state=active]:bg-slate-700 data-[state=active]:text-yellow-400 text-slate-400 cursor-pointer"
                style={{ pointerEvents: 'auto' }}
              >
                <Box className="w-4 h-4 mr-2" />
                3D Visualization
              </TabsTrigger>
            </TabsList>

            {/* Dashboard Content - Project Notebook */}
            <TabsContent value="notebook" className="mt-0 h-full overflow-hidden">
              <NotebookViewer
                title="GeoAuPredict - Project Presentation Notebook"
                description="Complete 3-phase pipeline: Data Ingestion → Feature Engineering → Predictive Modeling. Comprehensive notebook with detailed explanations and results."
              />
            </TabsContent>

            {/* Dashboard Content - Spatial Validation */}
            <TabsContent value="spatial" className="mt-0 h-full overflow-hidden">
              <div className="w-full h-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden">
                {showStreamlitHelp ? (
                  <div className="flex flex-col items-center justify-center h-full p-8 text-center">
                    <Map className="w-16 h-16 text-yellow-400 mb-4" />
                    <h3 className="text-2xl font-bold text-white mb-4">
                      Interactive Dashboards Available Locally
                    </h3>
                    <p className="text-slate-300 mb-6 max-w-2xl">
                      The interactive Streamlit dashboard requires a Python backend and is available when running locally.
                    </p>
                    <div className="bg-slate-800 p-6 rounded-lg max-w-2xl text-left">
                      <h4 className="text-lg font-semibold text-yellow-400 mb-3">To Run Locally:</h4>
                      <pre className="text-sm text-slate-300 bg-slate-900 p-4 rounded overflow-x-auto">
                         {`# Clone the repository
                         git clone https://github.com/edwardcalderon/GeoAuPredict.git
                         cd GeoAuPredict
 
                         # Install dependencies
                         pip install -r requirements_full.txt
 
                         # Start dashboards
                         ./start_dashboards.sh
 
                         # Open http://localhost:8501`}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <iframe
                    src={`${STREAMLIT_URL}?embed=true`}
                    className="w-full h-full border-0"
                    title="Spatial Validation Dashboard"
                    allow="fullscreen"
                    loading="lazy"
                  />
                )}
              </div>
            </TabsContent>

            {/* Dashboard Content - 3D Visualization (Static Plotly.js) */}
            <TabsContent value="3d" className="mt-0 h-full overflow-hidden">
              <Viz3D />
            </TabsContent>
            </Tabs>
          </div>
        </main>

        {/* Fixed Footer */}
        <div className="flex-shrink-0">
          <Footer />
        </div>
      </div>
    </ProtectedRoute>
  );
}
