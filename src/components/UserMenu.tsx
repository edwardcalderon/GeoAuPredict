'use client';

import { useState, useRef, useEffect } from 'react';
import { User, LogOut, ChevronDown } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export default function UserMenu() {
  const { user, signOut } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  if (!user) return null;

  // Get user initials from email
  const getInitials = (email: string) => {
    return email.substring(0, 2).toUpperCase();
  };

  // Get avatar color based on email
  const getAvatarColor = (email: string) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-purple-500',
      'bg-pink-500',
      'bg-indigo-500',
      'bg-yellow-500',
    ];
    const index = email.charCodeAt(0) % colors.length;
    return colors[index];
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* Avatar Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 p-1 rounded-lg hover:bg-slate-700/50 transition-colors"
      >
        <div className={`w-9 h-9 rounded-full ${getAvatarColor(user.email || '')} flex items-center justify-center text-white font-semibold text-sm`}>
          {getInitials(user.email || 'U')}
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-slate-900 border-2 border-slate-600 rounded-lg shadow-2xl overflow-hidden z-[99999]">
          {/* User Info */}
          <div className="p-4 border-b-2 border-slate-600 bg-slate-800">
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-full ${getAvatarColor(user.email || '')} flex items-center justify-center text-white font-semibold shadow-lg`}>
                {getInitials(user.email || 'U')}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white truncate">
                  {user.email}
                </p>
                <p className="text-xs text-slate-300">
                  Signed in
                </p>
              </div>
            </div>
          </div>

          {/* Menu Items */}
          <div className="p-2 bg-slate-900">
            <button
              onClick={() => {
                setIsOpen(false);
                signOut();
              }}
              className="w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium text-white hover:bg-red-600 hover:text-white rounded-md transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Sign out</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

