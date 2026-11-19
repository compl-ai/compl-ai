'use client';
import { useId, useState } from 'react';

// Pill tabs over server-rendered panels; hidden panels stay in the page for search and printing.
export function RunTabs({ label, tabs }: { label: string; tabs: { id: string; label: string; content: React.ReactNode }[] }) {
  const [active, setActive] = useState(tabs[0]?.id);
  const base = useId();
  return (
    <div>
      <div role="tablist" aria-label={label} className="inline-flex h-[34px] items-center rounded-full border border-border bg-card p-[3px]">
        {tabs.map(t => (
          <button
            key={t.id}
            id={`${base}-${t.id}-tab`}
            type="button"
            role="tab"
            aria-selected={t.id === active}
            aria-controls={`${base}-${t.id}`}
            onClick={() => setActive(t.id)}
            className={`h-full cursor-pointer rounded-full px-3 text-xs font-semibold transition-colors ${t.id === active ? 'bg-ink text-card' : 'text-ink/70 hover:text-ink'}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tabs.map(t => (
        <div key={t.id} id={`${base}-${t.id}`} role="tabpanel" aria-labelledby={`${base}-${t.id}-tab`} hidden={t.id !== active} className="mt-4">
          {t.content}
        </div>
      ))}
    </div>
  );
}
