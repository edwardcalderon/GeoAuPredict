import AppHeader from '@/components/AppHeader';
import Footer from '@/components/Footer';
import CloseableNotice from '@/components/CloseableNotice';
import { promises as fs } from 'fs';
import { join } from 'path';
import React  from 'react';
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
      <main className="flex-1 overflow-y-auto relative z-10">

        <div className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            <CloseableNotice versions={versions} />
            
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-8">
              {/* Jupyter Book Content */}
              <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
                <iframe
                  src={`${getNavUrl('/jupyter-book/intro.html')}`}
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
        <Footer author="Edward Calderón" version={versions.currentVersion} />
      </div>
    </div>
  );
}
