export const EYEBROW_CLASS = 'font-plex text-[13px] font-semibold uppercase tracking-[0.1em] text-brand';

export function Eyebrow({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return <p className={`${EYEBROW_CLASS} mb-1.5 ${className}`}>{children}</p>;
}
