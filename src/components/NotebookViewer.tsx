'use client';

import React from 'react';
import { ExternalLink, Book, Code, Download, FileText, Play, GitBranch } from 'lucide-react';

interface NotebookViewerProps {
  notebookUrl?: string;
  title?: string;
  description?: string;
}

export default function NotebookViewer({
  notebookUrl = 'https://github.com/edwardcalderon/GeoAuPredict/blob/main/notebooks/GeoAuPredict_Project_Presentation.ipynb',
  title = 'GeoAuPredict Project Presentation',
  description = 'Complete project review - Interactive notebook demonstrating the full 3-phase pipeline from data ingestion to predictive modeling.'
}: NotebookViewerProps) {
  const githubUrl = notebookUrl;
  const rawNotebookUrl = notebookUrl
    .replace('github.com', 'raw.githubusercontent.com')
    .replace('/blob/', '/');
  const colabUrl = `https://colab.research.google.com/github/edwardcalderon/GeoAuPredict/blob/main/notebooks/GeoAuPredict_Project_Presentation.ipynb`;

  return (
    <div className="w-full h-full flex flex-col bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="flex-1 p-4 md:p-6 lg:p-8 overflow-hidden">
        <div className="max-w-7xl mx-auto h-full">
          {/* Main Card */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 shadow-xl h-full flex flex-col overflow-hidden">
            {/* Header - Compact */}
            <div className="bg-gradient-to-r from-yellow-600/20 to-orange-600/20 border-b border-slate-700 p-4 md:p-6 flex-shrink-0">
              <div className="flex items-center gap-3">
                <Book className="w-8 h-8 md:w-10 md:h-10 text-yellow-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <h2 className="text-xl md:text-2xl font-bold text-white truncate">{title}</h2>
                  <p className="text-sm md:text-base text-slate-300 line-clamp-2">{description}</p>
                </div>
              </div>
            </div>

            {/* Content Sections - Scrollable */}
            <div className="flex-1 p-4 md:p-6 space-y-4 md:space-y-5 overflow-y-auto">
              {/* Notebook Contents Preview */}
              <div>
                <h3 className="text-lg md:text-xl font-semibold text-white mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4 md:w-5 md:h-5 text-yellow-400" />
                  Notebook Contents
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3">
                  {[
                    { num: '1', title: 'Introduction & Problem', icon: '🎯' },
                    { num: '2', title: 'Data Sources (6 sources)', icon: '🗄️' },
                    { num: '3', title: '3-Phase Architecture', icon: '🏗️' },
                    { num: '4', title: 'Phase 1: Data Ingestion', icon: '📥' },
                    { num: '5', title: 'Phase 2: Feature Engineering', icon: '🔧' },
                    { num: '6', title: 'Phase 3: Predictive Modeling', icon: '🤖' },
                    { num: '7', title: 'Results & Performance', icon: '📊' },
                    { num: '8', title: 'Interactive Demos', icon: '🗺️' },
                    { num: '9', title: 'Conclusions & Future Work', icon: '🚀' },
                  ].map((section) => (
                    <div
                      key={section.num}
                      className="flex items-center gap-2 p-2 bg-slate-700/30 rounded-lg border border-slate-600/50"
                    >
                      <span className="text-lg md:text-xl flex-shrink-0">{section.icon}</span>
                      <div className="min-w-0 flex-1">
                        <div className="text-[10px] md:text-xs text-slate-400">Section {section.num}</div>
                        <div className="text-xs md:text-sm font-medium text-white truncate">{section.title}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Key Highlights */}
              <div>
                <h3 className="text-lg md:text-xl font-semibold text-white mb-3 flex items-center gap-2">
                  <GitBranch className="w-4 h-4 md:w-5 md:h-5 text-yellow-400" />
                  Key Highlights
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3">
                  {[
                    { label: 'Data Sources', value: '6 integrated', color: 'blue' },
                    { label: 'Dataset Size', value: '10,247 samples', color: 'green' },
                    { label: 'Features', value: '35+ engineered', color: 'purple' },
                    { label: 'Model AUC', value: '0.87 (Excellent)', color: 'yellow' },
                    { label: 'Coverage', value: '1.14M km²', color: 'cyan' },
                    { label: 'Cost Reduction', value: '59% savings', color: 'red' },
                  ].map((item) => (
                    <div
                      key={item.label}
                      className="p-2 md:p-3 bg-slate-700/30 rounded-lg border border-slate-600/50"
                    >
                      <div className="text-[10px] md:text-xs text-slate-400 mb-0.5">{item.label}</div>
                      <div className="text-sm md:text-base font-bold text-yellow-400 truncate">{item.value}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div>
                <h3 className="text-lg md:text-xl font-semibold text-white mb-3 flex items-center gap-2">
                  <Play className="w-4 h-4 md:w-5 md:h-5 text-yellow-400" />
                  View & Run Notebook
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-3">
                  <a
                    href={githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-3 md:p-4 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <ExternalLink className="w-5 h-5 md:w-6 md:h-6 flex-shrink-0" />
                    <div className="text-left min-w-0">
                      <div className="text-sm md:text-base truncate">View on GitHub</div>
                      <div className="text-[10px] md:text-xs opacity-80 truncate">Read-only, rendered view</div>
                    </div>
                  </a>

                  <a
                    href={colabUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-3 md:p-4 bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <Play className="w-5 h-5 md:w-6 md:h-6 flex-shrink-0" />
                    <div className="text-left min-w-0">
                      <div className="text-sm md:text-base truncate">Run in Google Colab</div>
                      <div className="text-[10px] md:text-xs opacity-80 truncate">Interactive, requires Google account</div>
                    </div>
                  </a>

                  <a
                    href={rawNotebookUrl}
                    download
                    className="flex items-center gap-3 p-3 md:p-4 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <Download className="w-5 h-5 md:w-6 md:h-6 flex-shrink-0" />
                    <div className="text-left min-w-0">
                      <div className="text-sm md:text-base truncate">Download .ipynb</div>
                      <div className="text-[10px] md:text-xs opacity-80 truncate">Run locally in Jupyter</div>
                    </div>
                  </a>

                  <a
                    href={githubUrl.replace('/blob/', '/raw/')}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-3 md:p-4 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
                  >
                    <Code className="w-5 h-5 md:w-6 md:h-6 flex-shrink-0" />
                    <div className="text-left min-w-0">
                      <div className="text-sm md:text-base truncate">View Raw Code</div>
                      <div className="text-[10px] md:text-xs opacity-80 truncate">JSON source code</div>
                    </div>
                  </a>
                </div>
              </div>

              {/* Info Note */}
              <div className="p-3 md:p-4 bg-yellow-600/10 border border-yellow-600/30 rounded-lg">
                <div className="flex gap-2 md:gap-3">
                  <div className="flex-shrink-0 text-lg md:text-xl">💡</div>
                  <div className="text-xs md:text-sm text-slate-300">
                    <strong className="text-yellow-400">Recommended:</strong> Click "Run in Google Colab" 
                    for the best interactive experience. Execute code cells, modify parameters, 
                    and see live visualizations. Or view on GitHub for a static rendered version.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

