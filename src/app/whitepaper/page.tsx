import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { promises as fs } from 'fs';
import { join } from 'path';
import React  from 'react';
import ReactMarkdown from 'react-markdown';
import { Download } from 'lucide-react';
import { getNavUrl } from '@/lib/navigation';

export default async function WhitePaperPage() {
  // Read the markdown whitepaper content
  const whitepaperPath = join(process.cwd(), 'docs', 'whitepaper.md');
  const content = await fs.readFile(whitepaperPath, 'utf8');

  // Read version information
  const versionsPath = join(process.cwd(), 'public', 'versions', 'versions.json');
  const versionsData = await fs.readFile(versionsPath, 'utf8');
  const versions = JSON.parse(versionsData);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      <Header
        logoText="GAP"
        title="Geo Au Predict"
        navigation={[
          { label: 'Home', href: getNavUrl('/'), isActive: false },
          { label: 'White Paper', href: getNavUrl('/whitepaper'), isActive: true },
          { label: 'GitHub', href: 'https://github.com/edwardcalderon/GeoAuPredict', isActive: false, target: '_blank' }
        ]}
      />

      {/* Floating Download Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <a
          href={versions.downloadUrl}
          download={`GeoAuPredict_GAP_WhitePaper_${versions.currentVersion}.pdf`}
          className="flex items-center space-x-2 px-4 py-3 bg-yellow-600 hover:bg-yellow-700 text-slate-900 rounded-full font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          <Download className="w-5 h-5" />
          <span className="hidden sm:inline">Download PDF</span>
        </a>
      </div>

      {/* White Paper Content - This will grow to fill available space */}
      <main className="flex-grow">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-8">
              <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <h3 className="text-yellow-400 font-semibold mb-2">📝 Note on Mathematical Content</h3>
                <p className="text-slate-300 text-sm">
                  For the best viewing experience with mathematical formulas and equations, please download the PDF version.
                </p>
                <a
                  href={versions.downloadUrl}
                  download={`GeoAuPredict_GAP_WhitePaper_${versions.currentVersion}.pdf`}
                  className="inline-flex items-center mt-3 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-slate-900 rounded text-sm font-medium transition-colors"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download PDF
                </a>
              </div>
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown
                  components={{
                    // Custom styling for markdown elements
                    h1: ({ children }) => (
                      <h1 className="text-3xl font-bold text-white mb-6 border-b border-slate-600 pb-4">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-2xl font-semibold text-yellow-400 mt-8 mb-4">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-xl font-medium text-slate-200 mt-6 mb-3">
                        {children}
                      </h3>
                    ),
                    p: ({ children }) => (
                      <p className="text-slate-300 leading-relaxed mb-4">
                        {children}
                      </p>
                    ),
                    ul: ({ children }) => (
                      <ul className="text-slate-300 space-y-2 mb-4 ml-6">
                        {children}
                      </ul>
                    ),
                    li: ({ children }) => (
                      <li className="list-disc text-slate-300">
                        {children}
                      </li>
                    ),
                    strong: ({ children }) => (
                      <strong className="text-white font-semibold">
                        {children}
                      </strong>
                    ),
                    em: ({ children }) => (
                      <em className="text-slate-200 italic">
                        {children}
                      </em>
                    ),
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        className="text-yellow-400 hover:text-yellow-300 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {children}
                      </a>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-yellow-500 pl-4 italic text-slate-400 my-4">
                        {children}
                      </blockquote>
                    ),
                    code: ({ children }) => {
                      // Handle inline LaTeX/math expressions
                      const content = String(children);
                      if (content.includes('$') || content.includes('\\')) {
                        return (
                          <code className="bg-slate-700 text-slate-200 px-2 py-1 rounded text-sm font-mono">
                            [LaTeX: {content}]
                          </code>
                        );
                      }
                      return (
                        <code className="bg-slate-700 text-slate-200 px-2 py-1 rounded text-sm">
                          {children}
                        </code>
                      );
                    },
                    pre: ({ children }) => {
                      // Handle code blocks with LaTeX/math
                      // Properly handle different types of children (React elements vs strings)
                      let content = '';

                      if (React.isValidElement(children)) {
                        // If children is a React element, access its props.children
                        content = children.props?.children || '';
                      } else {
                        // If children is a string or other primitive, use it directly
                        content = String(children || '');
                      }

                      if (content.includes('$') || content.includes('\\')) {
                        return (
                          <div className="bg-slate-900 text-slate-200 p-4 rounded-lg my-4 border border-slate-600">
                            <div className="text-slate-400 text-sm mb-2">Mathematical Expression:</div>
                            <code className="text-slate-300 font-mono text-sm">
                              {content}
                            </code>
                            <div className="text-slate-400 text-xs mt-2">
                              💡 Download PDF for proper mathematical rendering
                            </div>
                          </div>
                        );
                      }
                      return (
                        <pre className="bg-slate-900 text-slate-200 p-4 rounded-lg overflow-x-auto my-4">
                          {children}
                        </pre>
                      );
                    },
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer - This will stick to the bottom */}
      <Footer />
    </div>
  );
}
