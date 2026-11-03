import localFont from 'next/font/local';
import { IBM_Plex_Mono, IBM_Plex_Sans } from 'next/font/google';

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
export const plexSans = IBM_Plex_Sans({ subsets: ['latin'], weight: ['400', '500', '600'], variable: '--font-plex-sans', display: 'swap' });
export const plexMono = IBM_Plex_Mono({ subsets: ['latin'], weight: ['400', '500', '600'], variable: '--font-plex-mono', display: 'swap' });
