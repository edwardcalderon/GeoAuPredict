'use client';

import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { Shield } from 'lucide-react';

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      // Redirect to login if not authenticated
      router.push('/login');
    }
  }, [user, loading, router]);

  // Show loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-yellow-400/10 mb-4">
            <Shield className="w-8 h-8 text-yellow-400 animate-pulse" />
          </div>
          <h2 className="text-xl font-semibold text-white mb-2">Verifying Access</h2>
          <p className="text-slate-400">Please wait...</p>
        </div>
      </div>
    );
  }

  // Show nothing while redirecting
  if (!user) {
    return null;
  }

  // User is authenticated, show the protected content
  return <>{children}</>;
}

