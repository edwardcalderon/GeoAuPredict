interface HeaderProps {
  logoText?: string;
  title: string;
  navigation?: Array<{
    label: string;
    href: string;
    isActive?: boolean;
  }>;
}

export default function Header({ logoText = "G", title, navigation = [] }: HeaderProps) {
  return (
    <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg flex items-center justify-center">
            <span className="text-slate-900 font-bold text-xs leading-none">{logoText}</span>
          </div>
          <h1 className="text-xl font-bold text-white">{title}</h1>
        </div>
        {navigation.length > 0 && (
          <nav className="flex items-center space-x-6">
            {navigation.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className={`transition-colors ${
                  item.isActive
                    ? "text-yellow-400 font-semibold"
                    : "text-slate-300 hover:text-white"
                }`}
              >
                {item.label}
              </a>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}
