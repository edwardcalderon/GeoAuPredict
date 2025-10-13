'use client';

import { useState } from 'react';
import { Download, X, Info } from 'lucide-react';
import { getNavUrl } from '@/lib/navigation';

interface CloseableNoticeProps {
  versions: {
    currentVersion: string;
    downloadUrl: string;
  };
}

export default function CloseableNotice({ versions }: CloseableNoticeProps) {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-24 right-6 z-[100] inline-flex items-center gap-2 px-4 py-2 bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/20 hover:border-yellow-500/30 rounded-lg text-yellow-400 hover:text-yellow-300 text-sm font-medium transition-all shadow-lg hover:shadow-yellow-500/20"
        aria-label="Show notice"
      >
        <Info className="w-4 h-4" />
        <span>Show Publication Info</span>
      </button>
    );
  }

  return (
    <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg relative">
      <button
        onClick={() => setIsVisible(false)}
        className="absolute top-3 right-3 text-yellow-400 hover:text-yellow-300 transition-colors"
        aria-label="Close notice"
      >
        <X className="w-5 h-5" />
      </button>
      
      <h3 className="text-yellow-400 font-semibold mb-2 pr-8">📚 Enhanced Scientific Publishing</h3>
      <p className="text-slate-300 text-sm mb-3">
        For the best viewing experience with mathematical formulas and equations, please download the PDF version.
      </p>
      <div className="flex gap-3">
        <a
          href={getNavUrl(versions.downloadUrl)}
          download={`GeoAuPredict_GAP_WhitePaper_${versions.currentVersion}.pdf`}
          className="inline-flex items-center px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-slate-900 rounded text-sm font-medium transition-colors"
        >
          <Download className="w-4 h-4 mr-2" />
          Download PDF
        </a>
        <a
          href={getNavUrl('/jupyter-book/intro.html')}
          className="inline-flex items-center px-3 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded text-sm font-medium transition-colors"
          target="_blank"
          rel="noopener noreferrer"
        >
          📖 View in Jupyter Book
        </a>
      </div>
    </div>
  );
}

