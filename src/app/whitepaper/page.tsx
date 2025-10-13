import AppHeader from '@/components/AppHeader';
import Footer from '@/components/Footer';
import { promises as fs } from 'fs';
import { join } from 'path';
import React  from 'react';
import { Download } from 'lucide-react';
import { getNavUrl } from '@/lib/navigation';

export default async function WhitePaperPage() {
  // Read version information for download links
  const versionsPath = join(process.cwd(), 'public', 'versions', 'whitepaper-version.json');
  const versionsData = await fs.readFile(versionsPath, 'utf8');
  const versions = JSON.parse(versionsData);


  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col overflow-hidden">
      {/* Fixed Header */}
      <div className="flex-shrink-0">
        <AppHeader currentPage="whitepaper" />
      </div>

      {/* White Paper Content - Scrollable */}
      <main className="flex-1 overflow-y-auto relative">

        <div className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-8">
              <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <h3 className="text-yellow-400 font-semibold mb-2">📚 Enhanced Scientific Publishing</h3>
                <p className="text-slate-300 text-sm">
                  This whitepaper is now published using Jupyter Book for optimal scientific content presentation with proper mathematical rendering, professional styling, and enhanced navigation.
                </p>
                <div className="flex gap-3 mt-3">
                  <a
                    href={versions.downloadUrl}
                    download={getNavUrl(`GeoAuPredict_GAP_WhitePaper_${versions.currentVersion}.pdf`)}
                    className="inline-flex items-center px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-slate-900 rounded text-sm font-medium transition-colors"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download PDF
                  </a>
                  <a
                    href="/jupyter-book/intro.html"
                    className="inline-flex items-center px-3 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded text-sm font-medium transition-colors"
                  >
                    📖 View in Jupyter Book
                  </a>
                </div>
              </div>
              {/* Jupyter Book Content */}
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
                <iframe
                  src="/jupyter-book/intro.html"
                  className="w-full h-[calc(100vh-300px)] border-0"
                  title="GeoAuPredict White Paper"
                  sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
                />
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Fixed Footer */}
      <div className="flex-shrink-0">
        <Footer />
      </div>
    </div>
  );
}
