'use client';

import { useAuth } from '@/contexts/AuthContext';
import Header from '@/components/Header';
import UserMenu from '@/components/UserMenu';
import { getNavUrl } from '@/lib/navigation';

interface AppHeaderProps {
  currentPage: 'home' | 'dashboards' | 'whitepaper';
}

export default function AppHeader({ currentPage }: AppHeaderProps) {
  const { user } = useAuth();

  return (
    <Header
      logoText="GAP"
      title="Geo Au Predict"
      logoHref={getNavUrl('/')}
      navigation={[
        { label: 'Dashboards', href: getNavUrl('/dashboards'), isActive: currentPage === 'dashboards' },
        { label: 'White Paper', href: getNavUrl('/whitepaper'), isActive: currentPage === 'whitepaper' },
        { label: 'GitHub', href: 'https://github.com/edwardcalderon/GeoAuPredict', isActive: false, target: '_blank' }
      ]}
    >
      {user && <UserMenu />}
    </Header>
  );
}

