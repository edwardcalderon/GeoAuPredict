'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { getNavUrl } from '@/lib/navigation';
import { Map, Box } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Dashboard URLs - use environment variables for production deployment
const isProduction = process.env.NODE_ENV === 'production';
const STREAMLIT_URL = isProduction ? process.env.NEXT_PUBLIC_STREAMLIT_URL : 'http://localhost:8501';
const DASH_URL = isProduction ? process.env.NEXT_PUBLIC_DASH_URL : 'http://localhost:8050';


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
                <iframe
                  src={`${STREAMLIT_URL}?embed=true`}
                  className="w-full h-full border-0"
                  title="Spatial Validation Dashboard"
                  allow="fullscreen"
                  loading="lazy"
                />
              </div>
            </TabsContent>

            {/* Dashboard Content - 3D Visualization */}
            <TabsContent value="3d" className="mt-0 relative z-10">
              <div 
                className="w-full bg-slate-900/30 rounded-lg border border-slate-700 overflow-hidden"
                style={{ height: 'calc(100vh - 250px)' }}
              >
                <iframe
                  src={DASH_URL}
                  className="w-full h-full border-0"
                  title="3D Visualization Dashboard"
                  allow="fullscreen"
                  loading="lazy"
                />
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </main>

      <Footer />
    </div>
  );
}
