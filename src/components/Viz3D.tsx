'use client';

import { useEffect, useState } from 'react';

export default function Viz3D() {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return (
      <div className="flex items-center justify-center h-[600px] bg-slate-900 rounded-lg border border-slate-700">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400 mb-4"></div>
          <p className="text-slate-400">Loading 3D Visualization...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[800px] bg-slate-900 rounded-lg overflow-hidden border border-slate-700">
      <iframe
        src="/3d-visualization.html"
        className="w-full h-full min-h-[800px]"
        style={{
          border: 'none',
          borderRadius: '0.5rem',
        }}
        title="3D Gold Probability Visualization"
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  );
}

