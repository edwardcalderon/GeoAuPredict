// Supabase client configuration
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('❌ Supabase environment variables are missing!');
  console.error('NEXT_PUBLIC_SUPABASE_URL:', supabaseUrl ? '✓ Set' : '✗ Missing');
  console.error('NEXT_PUBLIC_SUPABASE_ANON_KEY:', supabaseAnonKey ? '✓ Set' : '✗ Missing');
} else {
  console.log('✓ Supabase configured');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true,
  },
});

// Auth helper functions
export const authService = {
  // Sign in with Google OAuth
  async signInWithGoogle() {
    // Construct full redirect URL including basePath for GitHub Pages
    const basePath = process.env.NODE_ENV === 'production' ? '/GeoAuPredict' : '';
    const redirectUrl = typeof window !== 'undefined' 
      ? `${window.location.origin}${basePath}/dashboards`
      : `${basePath}/dashboards`;
    
    console.log('🔐 OAuth Redirect URL:', redirectUrl);
    
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: redirectUrl,
      },
    });
    
    return { data, error };
  },

  // Sign out
  async signOut() {
    const { error } = await supabase.auth.signOut();
    return { error };
  },

  // Get current session
  async getSession() {
    const { data, error } = await supabase.auth.getSession();
    return { data, error };
  },

  // Get current user
  async getUser() {
    const { data, error } = await supabase.auth.getUser();
    return { data, error };
  },
};

