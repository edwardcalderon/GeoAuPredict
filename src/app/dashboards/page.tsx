'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getNavUrl } from '@/lib/navigation';
import { Map, Box, ExternalLink, AlertCircle, Loader2 } from 'lucide-react';

// Dashboard URLs - use environment variables for production deployment
const STREAMLIT_URL = process.env.NEXT_PUBLIC_STREAMLIT_URL || 'http://localhost:8501';
const DASH_URL = process.env.NEXT_PUBLIC_DASH_URL || 'http://localhost:8050';

export default function DashboardsPage() {
  const [activeTab, setActiveTab] = useState('spatial');
  const [streamlitLoading, setStreamlitLoading] = useState(true);
  const [dashLoading, setDashLoading] = useState(true);
  const [streamlitRunning, setStreamlitRunning] = useState(false);
  const [dashRunning, setDashRunning] = useState(false);
  const [streamlitError, setStreamlitError] = useState(false);
  const [dashError, setDashError] = useState(false);

  // Reset loading states when tab changes
  useEffect(() => {
    if (activeTab === 'spatial' && !streamlitRunning && !streamlitError) {
      setStreamlitLoading(true);
      // Auto-hide loading after 8 seconds to prevent infinite loading
      const timeout = setTimeout(() => {
        setStreamlitLoading(false);
        setStreamlitRunning(true);
      }, 8000);
      return () => clearTimeout(timeout);
    }
    if (activeTab === '3d' && !dashRunning && !dashError) {
      setDashLoading(true);
      // Auto-hide loading after 8 seconds to prevent infinite loading
      const timeout = setTimeout(() => {
        setDashLoading(false);
        setDashRunning(true);
      }, 8000);
      return () => clearTimeout(timeout);
    }
  }, [activeTab, streamlitRunning, streamlitError, dashRunning, dashError]);

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
          {/* Compact Header with Status */}
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                Gold Prediction Analysis
              </h1>
              <p className="text-sm text-slate-400">
                Interactive spatial validation and 3D visualization
              </p>
            </div>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${streamlitRunning ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
                <span className="text-xs text-slate-400">Spatial</span>
                {streamlitRunning && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs hover:bg-slate-700"
                    onClick={() => window.open(STREAMLIT_URL, '_blank')}
                  >
                    <ExternalLink className="w-3 h-3" />
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${dashRunning ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
                <span className="text-xs text-slate-400">3D Viz</span>
                {dashRunning && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs hover:bg-slate-700"
                    onClick={() => window.open(DASH_URL, '_blank')}
                  >
                    <ExternalLink className="w-3 h-3" />
                  </Button>
                )}
              </div>
            </div>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-slate-800/50 border border-slate-700 mb-4">
              <TabsTrigger 
                value="spatial" 
                className="data-[state=active]:bg-slate-700 data-[state=active]:text-yellow-400 transition-all"
              >
                <Map className="w-4 h-4 mr-2" />
                Spatial Validation
              </TabsTrigger>
              <TabsTrigger 
                value="3d" 
                className="data-[state=active]:bg-slate-700 data-[state=active]:text-yellow-400 transition-all"
              >
                <Box className="w-4 h-4 mr-2" />
                3D Visualization
              </TabsTrigger>
            </TabsList>

            {/* Spatial Validation Dashboard */}
            <TabsContent value="spatial" className="mt-0">
              <div 
                className="relative w-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden"
                style={{ height: 'calc(100vh - 250px)' }}
              >
                {/* Loading State */}
                {streamlitLoading && !streamlitError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 z-10">
                    <div className="text-center space-y-4">
                      <Loader2 className="w-12 h-12 text-yellow-500 animate-spin mx-auto" />
                      <div>
                        <h3 className="text-lg font-semibold text-white">
                          Loading Spatial Validation Dashboard
                        </h3>
                        <p className="text-sm text-slate-400 mt-2">
                          Connecting to Streamlit on port 8501...
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {streamlitError && (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center space-y-4 p-8">
                      <AlertCircle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
                      <div>
                        <h3 className="text-xl font-semibold text-white mb-2">
                          Streamlit Dashboard Not Running
                        </h3>
                        <p className="text-sm text-slate-400 mb-4 max-w-md mx-auto">
                          Start the development server to view spatial validation results
                        </p>
                        <div className="bg-slate-800 rounded-lg p-4 mb-4">
                          <code className="text-sm text-green-400">
                            npm run dev:full
                          </code>
                        </div>
                        <p className="text-xs text-slate-500">
                          Or: <code className="text-slate-400">streamlit run src/app/spatial_validation_dashboard.py</code>
                        </p>
                        <Button
                          className="mt-4 bg-yellow-600 hover:bg-yellow-700"
                          onClick={() => {
                            setStreamlitError(false);
                            setStreamlitLoading(true);
                            // Force iframe reload
                            setTimeout(() => setStreamlitLoading(false), 100);
                          }}
                        >
                          Retry Connection
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Iframe */}
                {!streamlitError && (
                  <iframe
                    src={`${STREAMLIT_URL}?embed=true`}
                    className="w-full h-full border-0"
                    title="Spatial Validation Dashboard"
                    allow="fullscreen"
                    onLoad={() => {
                      console.log('Streamlit iframe loaded successfully');
                      setTimeout(() => {
                        setStreamlitLoading(false);
                        setStreamlitError(false);
                        setStreamlitRunning(true);
                      }, 1000);
                    }}
                  />
                )}
              </div>
            </TabsContent>

            {/* 3D Visualization Dashboard */}
            <TabsContent value="3d" className="mt-0">
              <div 
                className="relative w-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden"
                style={{ height: 'calc(100vh - 250px)' }}
              >
                {/* Loading State */}
                {dashLoading && !dashError && (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 z-10">
                    <div className="text-center space-y-4">
                      <Loader2 className="w-12 h-12 text-yellow-500 animate-spin mx-auto" />
                      <div>
                        <h3 className="text-lg font-semibold text-white">
                          Loading 3D Visualization Dashboard
                        </h3>
                        <p className="text-sm text-slate-400 mt-2">
                          Connecting to Dash on port 8050...
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {dashError && (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center space-y-4 p-8">
                      <AlertCircle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
                      <div>
                        <h3 className="text-xl font-semibold text-white mb-2">
                          3D Dashboard Not Running
                        </h3>
                        <p className="text-sm text-slate-400 mb-4 max-w-md mx-auto">
                          Start the development server to view 3D terrain visualization
                        </p>
                        <div className="bg-slate-800 rounded-lg p-4 mb-4">
                          <code className="text-sm text-green-400">
                            npm run dev:full
                          </code>
                        </div>
                        <p className="text-xs text-slate-500">
                          Or: <code className="text-slate-400">python src/app/3d_visualization_dashboard.py</code>
                        </p>
                        <Button
                          className="mt-4 bg-yellow-600 hover:bg-yellow-700"
                          onClick={() => {
                            setDashError(false);
                            setDashLoading(true);
                            // Force iframe reload
                            setTimeout(() => setDashLoading(false), 100);
                          }}
                        >
                          Retry Connection
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Iframe */}
                {!dashError && (
                  <iframe
                    src={DASH_URL}
                    className="w-full h-full border-0"
                    title="3D Visualization Dashboard"
                    allow="fullscreen"
                    onLoad={() => {
                      console.log('Dash iframe loaded successfully');
                      setTimeout(() => {
                        setDashLoading(false);
                        setDashError(false);
                        setDashRunning(true);
                      }, 1000);
                    }}
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
