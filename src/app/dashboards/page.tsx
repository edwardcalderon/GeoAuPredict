'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { getNavUrl } from '@/lib/navigation';
import { Map, Box } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Dashboard URLs - use environment variables for production deployment
const isProduction = process.env.NODE_ENV === 'production';
const STREAMLIT_URL = process.env.NEXT_PUBLIC_STREAMLIT_URL || 'http://localhost:8501';
const DASH_URL = process.env.NEXT_PUBLIC_DASH_URL || 'http://localhost:8050';

// Check if we have a valid deployed dashboard URL (not localhost)
const hasDeployedStreamlit = STREAMLIT_URL && !STREAMLIT_URL.includes('localhost');
const hasDeployedDash = DASH_URL && !DASH_URL.includes('localhost');

// Show helpful message only if we don't have a deployed dashboard
const showStreamlitHelp = isProduction && !hasDeployedStreamlit;
const showDashHelp = isProduction && !hasDeployedDash;


export default function DashboardsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      <Header
        logoText="GAP"
        title="Geo Au Predict"
        navigation={[
          { label: 'Home', href: getNavUrl('/'), isActive: false },
          { label: 'Dashboards', href: getNavUrl('/dashboards'), isActive: true },
          { label: 'White Paper', href: getNavUrl('/whitepaper'), isActive: false },
          { label: 'GitHub', href: 'https://github.com/edwardcalderon/GeoAuPredict', isActive: false }
        ]}
      />

      <main className="flex-grow container mx-auto px-4 py-4">
        <div className="max-w-full mx-auto">
          {/* Header */}
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                Gold Prediction Analysis
              </h1>
              <p className="text-sm text-slate-400">
                Interactive spatial validation and 3D visualization
              </p>
            </div>
          </div>

          {/* Tabs Component */}
          <Tabs defaultValue="spatial" className="w-full relative z-50">
            <TabsList className="grid w-full grid-cols-2 bg-slate-800/50 border border-slate-700 rounded-lg mb-4 relative z-50" style={{ pointerEvents: 'auto' }}>
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

            {/* Dashboard Content - Spatial Validation */}
            <TabsContent value="spatial" className="mt-0 relative z-10">
              <div
                className="w-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden"
                style={{ height: 'calc(100vh - 250px)' }}
              >
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

            {/* Dashboard Content - 3D Visualization */}
            <TabsContent value="3d" className="mt-0 relative z-10">
              <div
                className="w-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden"
                style={{ height: 'calc(100vh - 250px)' }}
              >
                {showDashHelp ? (
                  <div className="flex flex-col items-center justify-center h-full p-8 text-center">
                    <Box className="w-16 h-16 text-yellow-400 mb-4" />
                    <h3 className="text-2xl font-bold text-white mb-4">
                      3D Visualization Available Locally
                    </h3>
                    <p className="text-slate-300 mb-6 max-w-2xl">
                      The interactive 3D Dash dashboard requires a Python backend and is available when running locally.
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
 
                           # Open http://localhost:8050`}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <iframe
                    src={DASH_URL}
                    className="w-full h-full border-0"
                    title="3D Visualization Dashboard"
                    allow="fullscreen"
                    loading="lazy"
                  />
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>

      <Footer />
    </div>
  );
}
