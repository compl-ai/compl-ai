import Image from 'next/image';
import { withBasePath } from '@/lib/utils';

// The "I" at the end of the wordmark spans y 7.26–21 in the 27-unit-tall viewBox, so its centre sits slightly below the logo's centre.
export const LOGO_I_CENTER_OFFSET = (7.26 + 21) / 2 / 27 - 0.5;

export const LOGO_HEIGHT = { md: 28.6, sm: 22 } as const;

export function BrandMark({ size = 'md' }: { size?: keyof typeof LOGO_HEIGHT }) {
  const height = LOGO_HEIGHT[size];
  return (
    <Image
      src={withBasePath('/assets/complai-logo.svg')}
      alt="ComplAI"
      width={Math.round((height * 103) / 27)}
      height={Math.round(height)}
      style={{ height, width: 'auto' }}
      className="block shrink-0 max-w-none"
      priority
      unoptimized
    />
  );
}

export function GitHubIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.58.23 2.75.11 3.04.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}
