import Link from 'next/link';

export function SiteFooter() {
  return (
    <footer className="border-t py-6 md:py-0">
      <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row px-4 md:px-8 max-w-7xl mx-auto">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0 text-sm leading-loose text-muted-foreground">
          <p>
            Built by the <span className="font-semibold">COMPL-AI Team</span>. The source code is available on GitHub.
          </p>
        </div>
      </div>
    </footer>
  );
}
