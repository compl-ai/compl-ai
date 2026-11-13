import type { Metadata } from "next";
import "./globals.css";
import { awelier, plexMono, plexSans } from "./fonts";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { ScrollToTop } from "@/components/layout/ScrollToTop";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DARK_MODE } from "@/lib/theme";
import { SITE_DESCRIPTION, SITE_NAME, SITE_ORIGIN, SITE_TITLE, siteSocial } from "@/lib/site";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_ORIGIN),
  title: { default: SITE_TITLE, template: `%s · ${SITE_NAME}` },
  description: SITE_DESCRIPTION,
  ...siteSocial(SITE_TITLE, SITE_DESCRIPTION),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${awelier.variable} ${plexSans.variable} ${plexMono.variable} scroll-smooth`} data-scroll-behavior="smooth" data-dark-mode={DARK_MODE || undefined}>
      <body className="min-h-screen bg-background font-sans antialiased flex flex-col">
        <TooltipProvider>
          <ScrollToTop />
          <SiteHeader />
          <main className="flex-1">
            {children}
          </main>
          <SiteFooter />
        </TooltipProvider>
      </body>
    </html>
  );
}
