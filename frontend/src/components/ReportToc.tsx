'use client';
import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

export interface TocItem {
  id: string;
  number: string;
  label: string;
  depth: 0 | 1 | 2;
}

// Sticky contents for the report chapters; the entry whose heading last scrolled past the header is marked.
export function ReportToc({ items }: { items: TocItem[] }) {
  const [active, setActive] = useState(items[0]?.id);

  useEffect(() => {
    const update = () => {
      let current = items[0]?.id;
      for (const item of items) {
        const el = document.getElementById(item.id);
        if (el && el.getBoundingClientRect().top < 160) current = item.id;
      }
      setActive(current);
    };
    update();
    window.addEventListener('scroll', update, { passive: true });
    return () => window.removeEventListener('scroll', update);
  }, [items]);

  return (
    <nav aria-label="Contents" className="sticky top-28">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Contents</div>
      <ol className="mt-3 space-y-1.5 border-l border-border">
        {items.map(item => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              aria-current={item.id === active ? 'location' : undefined}
              className={cn(
                '-ml-px flex gap-2 border-l py-0.5 leading-snug transition-colors hover:text-ink',
                item.depth === 0 ? 'pl-3 text-[13px] font-semibold' : item.depth === 1 ? 'pl-6 text-xs' : 'pl-9 text-xs',
                item.id === active ? 'border-ink text-ink' : 'border-transparent text-muted-foreground',
              )}
            >
              <span className="tabular text-muted-foreground/70">{item.number}</span>
              <span>{item.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
