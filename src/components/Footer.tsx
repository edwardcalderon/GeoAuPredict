export default function Footer() {
  return (
    <footer className="border-t border-slate-700 bg-slate-900/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded flex items-center justify-center">
              <span className="text-slate-900 font-bold text-xs">G</span>
            </div>
            <span className="text-slate-400">© 2025 GeoAuPredict GAP. All rights reserved.</span>
          </div>
          <div className="text-slate-400 text-sm">
            Open-source AI for mineral exploration
          </div>
        </div>
      </div>
    </footer>
  );
}
