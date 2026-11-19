import { withBasePath } from './utils';

export const REPO_URL = 'https://github.com/compl-ai/compl-ai';
export const README_CORE_URL = `${REPO_URL}#complai-core`;
export const README_QUICKSTART_URL = `${REPO_URL}#-quickstart`;

export const SITE_NAME = 'ComplAI CORE';
export const SITE_TITLE = `${SITE_NAME} - Efficient, open, and reproducible AI evaluation`;
export const SITE_DESCRIPTION = 'Open, calibrated AI evaluations that predict broad benchmark performance from a compact set of tasks.';
// Site origin for absolute social-preview URLs; the Pages workflow passes the full site URL, and basePath is added separately.
export const SITE_ORIGIN = new URL(process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000').origin;
const OG_IMAGE = { url: withBasePath('/assets/og-image.png'), width: 1200, height: 630, alt: SITE_TITLE };

export const siteSocial = (title: string, description: string) => ({
  openGraph: { type: 'website' as const, siteName: SITE_NAME, title, description, images: [OG_IMAGE] },
  twitter: { card: 'summary_large_image' as const, title, description, images: [OG_IMAGE] },
});

// Per-page title and description, repeated for link previews (a page's openGraph replaces the layout's).
export function pageMetadata(title: string, description: string) {
  return { title, description, ...siteSocial(`${title} · ${SITE_NAME}`, description) };
}
