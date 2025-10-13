'use client';

import { usePathname } from 'next/navigation';
import { getNavUrl } from '@/lib/navigation';
interface HeaderProps {
  logoText?: string;
  title: string;
  logoHref?: string;
  navigation?: Array<{
    label: string;
    href: string;
    target?: string;
    isActive?: boolean;
  }>;
  children?: React.ReactNode;
}

export default function Header({ logoText = "G", title, logoHref, navigation = [], children }: HeaderProps) {
  const pathname = usePathname();
  const isTitleActive = logoHref && pathname === logoHref;

  return (
    <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {logoHref ? (
            <a href={logoHref} className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg flex items-center justify-center hover:opacity-80 transition-opacity">
              <img src={getNavUrl(`/logo.png`)} alt="Logo" className="w-9 h-9" />
            </a>
          ) : (
            <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg flex items-center justify-center">
              <img src={getNavUrl(`/logo.png`)} alt="Logo" className="w-9 h-9" />
            </div>
          )}
          {logoHref ? (
            <a href={logoHref} className={`text-xl font-bold transition-colors ${
              isTitleActive
                ? "text-yellow-400 font-semibold"
                : "text-white hover:text-yellow-400"
            }`}>
              {title}
            </a>
          ) : (
            <h1 className="text-xl font-bold text-white">{title}</h1>
          )}
        </div>
        <div className="flex items-center space-x-6">
          {navigation.length > 0 && (
            <nav className="flex items-center space-x-6">
              {navigation.map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  target={item.target}
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
          {children}
        </div>
      </div>
    </header>
  );
}
