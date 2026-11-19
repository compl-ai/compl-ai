'use client';
import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

export function CopyCommand({ lines }: { lines: string[] }) {
  const [copied, setCopied] = useState(false);
  const text = lines.filter(l => !l.startsWith('#')).join('\n');
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="relative min-w-0 rounded-xl bg-navy text-white dark:ring-1 dark:ring-border">
      <pre className="overflow-x-auto px-4 py-3.5 pr-12 font-mono text-[12px] leading-relaxed">
        {lines.map((l, i) => (
          <div key={i} className={l.startsWith('#') ? 'text-white/50' : ''}>{l}</div>
        ))}
      </pre>
      <button
        onClick={copy}
        aria-label="Copy command"
        className="absolute right-2.5 top-2.5 rounded-md p-1.5 text-white/60 hover:text-white hover:bg-white/10 transition-colors"
      >
        {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
      </button>
    </div>
  );
}
