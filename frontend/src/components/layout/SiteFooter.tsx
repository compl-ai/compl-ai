import Link from 'next/link';
import { BrandMark } from '@/components/layout/BrandMark';
import { REPO_URL } from '@/lib/site';

export function SiteFooter() {
  return (
    <footer className="border-t border-border bg-white">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between px-4 md:px-8 py-8 max-w-[1400px] mx-auto text-sm text-muted-foreground">
        <div className="flex items-center gap-4">
          <BrandMark size="sm" />
          <span>Open, reproducible and efficient AI evaluation.</span>
        </div>
        <div className="flex items-center gap-6">
          <Link href="/methodology" className="hover:text-ink transition-colors">Methodology</Link>
          <Link href="/about" className="hover:text-ink transition-colors">About</Link>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="hover:text-ink transition-colors">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
