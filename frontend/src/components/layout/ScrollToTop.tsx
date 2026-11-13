'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';

// Next.js scrolls a new page's content (below the header) into view rather than the window to the top,
// which leaves the page scrolled down by the header's height.
export function ScrollToTop() {
  const pathname = usePathname();
  useEffect(() => {
    if (!window.location.hash) window.scrollTo({ top: 0, behavior: 'instant' });
  }, [pathname]);
  return null;
}
