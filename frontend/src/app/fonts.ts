import localFont from 'next/font/local';

// MADE Awelier (MadeType) under a single-site web font licence. Only these WOFF2 conversions are committed;
// the original OTFs live in the gitignored frontend/font-source/ and must not be published or pushed.
export const awelier = localFont({
  src: [
    { path: './fonts/MADEAwelier-Light.woff2', weight: '300', style: 'normal' },
    { path: './fonts/MADEAwelier-Regular.woff2', weight: '400', style: 'normal' },
    { path: './fonts/MADEAwelier-Medium.woff2', weight: '500', style: 'normal' },
    { path: './fonts/MADEAwelier-Bold.woff2', weight: '700', style: 'normal' },
    { path: './fonts/MADEAwelier-ExtraBold.woff2', weight: '800', style: 'normal' },
    { path: './fonts/MADEAwelier-Black.woff2', weight: '900', style: 'normal' },
  ],
  variable: '--font-awelier',
  display: 'swap',
});

// Secondary sans and mono, used for small UI labels (version pill, credits, commands).
// Self-hosted latin subsets from Fontsource (OFL-1.1, see fonts/IBMPlex-OFL.txt): next/font/google breaks under
// Turbopack when Google serves `/l/font?kit=` URLs, and this also keeps builds offline.
export const plexSans = localFont({
  src: [
    { path: './fonts/IBMPlexSans-400.woff2', weight: '400', style: 'normal' },
    { path: './fonts/IBMPlexSans-500.woff2', weight: '500', style: 'normal' },
    { path: './fonts/IBMPlexSans-600.woff2', weight: '600', style: 'normal' },
  ],
  variable: '--font-plex-sans',
  display: 'swap',
});
export const plexMono = localFont({
  src: [
    { path: './fonts/IBMPlexMono-400.woff2', weight: '400', style: 'normal' },
    { path: './fonts/IBMPlexMono-500.woff2', weight: '500', style: 'normal' },
    { path: './fonts/IBMPlexMono-600.woff2', weight: '600', style: 'normal' },
  ],
  variable: '--font-plex-mono',
  display: 'swap',
});
