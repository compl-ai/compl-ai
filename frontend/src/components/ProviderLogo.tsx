import { Bot } from 'lucide-react';
import { cn, withBasePath } from '@/lib/utils';
import providerData from '../../public/data/providers.json';

type Provider = { logo: string; color: string | null };
const PROVIDERS: Record<string, Provider> = providerData.providers;
const DIR = '/assets/provider-logos/';

// Decorative: always render next to the organization name, which carries the meaning.
export function ProviderLogo({ org, variant = 'color', className }: { org: string | null | undefined; variant?: 'color' | 'mono'; className?: string }) {
  const provider = org ? PROVIDERS[org] : undefined;
  const size = cn('size-4 shrink-0', className);
  if (!provider) return <Bot className={size} strokeWidth={1.75} aria-hidden />;
  if (variant === 'color' && provider.color) {
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={withBasePath(DIR + provider.color)} alt="" aria-hidden className={cn(size, 'object-contain')} />;
  }
  // Single-colour marks are drawn as a mask so they take the current text colour.
  const url = `url(${withBasePath(DIR + provider.logo)})`;
  return (
    <span
      aria-hidden
      className={cn(size, 'inline-block bg-current')}
      style={{ maskImage: url, WebkitMaskImage: url, maskSize: 'contain', WebkitMaskSize: 'contain', maskRepeat: 'no-repeat', WebkitMaskRepeat: 'no-repeat', maskPosition: 'center', WebkitMaskPosition: 'center' }}
    />
  );
}
