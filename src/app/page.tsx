import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Database, Map, Brain, BarChart3, Shield, BookOpen } from 'lucide-react';
import { getNavUrl } from '@/lib/navigation';

export default function Page() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      <Header
        logoText="GAP"
        title="Geo Au Predict"
        navigation={[
          { label: 'Home', href: getNavUrl('/'), isActive: true },
          { label: 'White Paper', href: getNavUrl('/whitepaper'), isActive: false },
          { label: 'GitHub', href: 'https://github.com/edwardcalderon/GeoAuPredict', isActive: false }
        ]}
      />

      {/* Main content - This will grow to fill available space */}
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="container mx-auto px-4 py-16 text-center">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-5xl font-bold text-white mb-6">
              Geospatial Intelligence for
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-yellow-600"> Gold Prediction</span>
            </h2>
            <p className="text-xl text-slate-300 mb-8 leading-relaxed">
              Advanced AI-powered system that integrates multiple geospatial data sources to predict gold presence in subsoil with interactive 3D visualization and confidence levels for mining exploration.
            </p>
            <div className="flex gap-4 justify-center">
              <Button size="lg" className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-slate-900 font-semibold">
                Start Exploration
              </Button>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-white mb-4">Comprehensive Mining Intelligence</h3>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Leverage cutting-edge AI and geospatial technology to make informed mining decisions with unprecedented accuracy.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <Database className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">Data Integration Dashboard</CardTitle>
                <CardDescription className="text-slate-400">
                  Upload and normalize geochemical, geophysical, drilling, and satellite data with automatic CRS conversion
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <Brain className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">AI Prediction Engine</CardTitle>
                <CardDescription className="text-slate-400">
                  Multi-model approach combining XGBoost, LSTM, and 3D CNN with uncertainty quantification
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <Map className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">Interactive 3D Visualization</CardTitle>
                <CardDescription className="text-slate-400">
                  Deck.gl powered maps showing gold probability by depth with filtering and export options
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <BarChart3 className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">Model Metrics Panel</CardTitle>
                <CardDescription className="text-slate-400">
                  Complete performance analytics with precision, recall, and confidence intervals
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <Shield className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">Audit & Reproducibility</CardTitle>
                <CardDescription className="text-slate-400">
                  Data lineage tracking and complete reproducibility reports for regulatory compliance
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-colors">
              <CardHeader>
                <BookOpen className="w-10 h-10 text-yellow-500 mb-2" />
                <CardTitle className="text-white">Research White Paper</CardTitle>
                <CardDescription className="text-slate-400">
                  Comprehensive technical documentation with mathematical formulations and scientific methodology
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>

        {/* CTA Section */}
        <section className="container mx-auto px-4 py-16">
          <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl p-8 text-center border border-slate-600">
            <h3 className="text-3xl font-bold text-white mb-4">Ready to Transform Your Mining Operations?</h3>
            <p className="text-slate-300 mb-6 max-w-2xl mx-auto">
              Join leading mining companies using GeoGold AI to discover new opportunities and optimize exploration strategies.
            </p>
            <Button size="lg" className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-slate-900 font-semibold">
              Get Started Today
            </Button>
          </div>
        </section>
      </main>

      {/* Footer - This will stick to the bottom */}
      <Footer />
    </div>
  );
}