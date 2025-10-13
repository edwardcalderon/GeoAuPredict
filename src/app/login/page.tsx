'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Lock, AlertCircle } from 'lucide-react';
import { getNavUrl } from '@/lib/navigation';
import { supabase } from '@/lib/supabase';

export default function LoginPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGoogleSignIn = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/dashboards`,
        },
      });
      
      if (error) {
        throw error;
      }
      
    } catch (err: any) {
      console.error('Error:', err);
      setError(err.message || 'Failed to sign in with Google. Check .env.local');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
      <Header
        logoText="GAP"
        title="Geo Au Predict"
        logoHref={getNavUrl('/')}
        navigation={[
          { label: 'Dashboards', href: getNavUrl('/dashboards'), isActive: false },
          { label: 'White Paper', href: getNavUrl('/whitepaper'), isActive: false },
          { label: 'GitHub', href: 'https://github.com/edwardcalderon/GeoAuPredict', isActive: false, target: '_blank' }
        ]}
      />

      <main className="flex-grow flex items-center justify-center px-4 py-12">
        <Card className="w-full max-w-md bg-slate-800/50 border-slate-700">
          <CardHeader className="space-y-1">
            <div className="flex items-center justify-center mb-4">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-yellow-400 to-yellow-600 flex items-center justify-center">
                <Lock className="w-6 h-6 text-slate-900" />
              </div>
            </div>
            <CardTitle className="text-2xl text-center text-white">
              Welcome to GeoAuPredict
            </CardTitle>
            <CardDescription className="text-center text-slate-400">
              Sign in with your Google account to access the gold prediction dashboard
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/50 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Google Sign In Button */}
            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={isLoading}
              className="w-full bg-white hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 font-semibold flex items-center justify-center gap-3 py-4 px-6 rounded-lg transition-colors"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                  <span>Redirecting to Google...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path
                      fill="currentColor"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="currentColor"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="currentColor"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                      fill="currentColor"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                  </svg>
                  <span>Continue with Google</span>
                </>
              )}
            </button>

            <div className="text-center pt-4 border-t border-slate-700">
              <p className="text-sm text-slate-400">
                By continuing, you agree to our Terms of Service and Privacy Policy
              </p>
            </div>
          </CardContent>
        </Card>
      </main>

      <Footer />
    </div>
  );
}
