'use client';

import { useEffect, useState } from 'react';

export default function JupyterLiteViewer() {
  const [isMounted, setIsMounted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsMounted(true);
    // Simulate loading delay
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  if (!isMounted) {
    return (
      <div className="flex items-center justify-center h-[600px] bg-slate-900 rounded-lg border border-slate-700">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400 mb-4"></div>
          <p className="text-slate-400">Initializing JupyterLite...</p>
        </div>
      </div>
    );
  }

  // Encode the notebook URL
  const notebookUrl = encodeURIComponent(
    'https://raw.githubusercontent.com/edwardcalderon/GeoAuPredict/main/notebooks/GeoAuPredict_Project_Presentation.ipynb'
  );
  
  // Use JupyterLite from CDN with our notebook
  const jupyterLiteUrl = `https://jupyterlite.github.io/demo/repl/index.html?fromURL=${notebookUrl}`;

  return (
    <div className="w-full h-full min-h-[800px] bg-slate-900 rounded-lg overflow-hidden border border-slate-700 relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900 z-10">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-400 mb-4"></div>
            <p className="text-slate-300 text-lg font-semibold">Loading JupyterLite Environment...</p>
            <p className="text-slate-400 text-sm mt-2">This may take a few moments on first load</p>
          </div>
        </div>
      )}
      
      <iframe
        src={jupyterLiteUrl}
        className="w-full h-full min-h-[800px]"
        style={{
          border: 'none',
          borderRadius: '0.5rem',
        }}
        title="JupyterLite Notebook Viewer"
        sandbox="allow-scripts allow-same-origin allow-downloads allow-modals allow-forms"
        allow="cross-origin-isolated"
        onLoad={() => setIsLoading(false)}
      />
      
      <div className="absolute bottom-4 right-4 bg-slate-800/90 backdrop-blur-sm px-4 py-2 rounded-lg border border-slate-700 text-xs text-slate-300">
        💡 <span className="font-semibold">Tip:</span> This notebook runs entirely in your browser!
      </div>
    </div>
  );
}

