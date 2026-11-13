'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BrandMark, GitHubIcon, LOGO_HEIGHT, LOGO_I_CENTER_OFFSET } from '@/components/layout/BrandMark';

import { REPO_URL } from '@/lib/site';

const NAV = [
  { href: '/#results', label: 'Results', match: '/' },
  { href: '/models', label: 'Models', match: '/models' },
  { href: '/benchmarks', label: 'Benchmarks', match: '/benchmarks' },
  { href: '/mapping', label: 'AI Act mapping', match: '/mapping' },
  { href: '/methodology', label: 'How it Works', match: '/methodology' },
];

const CREATORS = [
  { name: 'ETH Zurich', href: 'https://www.sri.inf.ethz.ch/' },
  { name: 'INSAIT', href: 'https://insait.ai/' },
  { name: 'LatticeFlow AI', href: 'https://latticeflow.ai/' },
];

export function SiteHeader() {
  const pathname = usePathname();
  const isActive = (match: string) => match === '/' ? pathname === '/' : pathname.startsWith(match);

  const navLinks = NAV.map(({ href, label, match }) => (
    <Link
      key={href}
      href={href}
      className={`whitespace-nowrap font-serif font-medium transition-colors ${isActive(match) ? 'text-ink' : 'text-ink/70 hover:text-ink'}`}
    >
      {label}
    </Link>
  ));

  const createdBy = (
    <>
      Created by{' '}
      {CREATORS.map((c, i) => (
        <span key={c.name}>
          <a href={c.href} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-link">{c.name}</a>
          {i < CREATORS.length - 2 ? ', ' : i === CREATORS.length - 2 ? ', and ' : ''}
        </span>
      ))}
    </>
  );

  // Phones: only the link row sticks; the logo row scrolls away.
  return (
    <>
    <header className="sm:sticky top-0 z-50 w-full bg-card/90 backdrop-blur pt-3 md:pt-4 print:pt-0 print:bg-transparent print:backdrop-blur-none">
      <div className="frame h-16 grid grid-cols-[auto_1fr_auto] items-center gap-6">
        <div className="flex items-center gap-2.5">
          <Link href="/" aria-label="ComplAI CORE home" className="flex shrink-0"><BrandMark /></Link>
          <span
            className="whitespace-nowrap font-mono text-[11px] font-semibold text-ink bg-brand/30 dark:bg-brand/70 dark:text-navy rounded-full px-2.5 py-1 leading-none"
            style={{ transform: `translateY(${(LOGO_HEIGHT.md * LOGO_I_CENTER_OFFSET).toFixed(2)}px)` }}
            title="Current release of the CORE evaluation set and results"
          >
            CORE v1
          </span>
        </div>

        <nav className="hidden xl:flex print:hidden justify-center gap-10 text-[17px]" aria-label="Main">{navLinks}</nav>

        <div className="flex items-center justify-end gap-5 print:hidden">
          <p className="hidden md:block font-plex text-[11px] font-medium text-ink">{createdBy}</p>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full bg-ink text-background font-mono text-sm font-semibold px-3.5 py-1.5 hover:bg-ink/85 transition-colors whitespace-nowrap"
          >
            <GitHubIcon />
            Run CORE
          </a>
        </div>
      </div>
      <p className="md:hidden print:hidden frame -mt-1 font-plex text-[11px] font-medium text-ink">{createdBy}</p>
      <nav className="hidden sm:flex xl:hidden print:hidden frame justify-between md:justify-start gap-3 md:gap-8 overflow-x-auto pt-2.5 md:pt-0 pb-3 text-[13px] md:text-[16px]" aria-label="Main">{navLinks}</nav>
    </header>
    <nav className="sm:hidden print:hidden sticky top-0 z-50 w-full bg-card/90 backdrop-blur" aria-label="Main">
      <div className="frame flex justify-between gap-3 overflow-x-auto pt-2.5 pb-3 text-[12px] min-[400px]:text-[13px]">{navLinks}</div>
    </nav>
    </>
  );
}
