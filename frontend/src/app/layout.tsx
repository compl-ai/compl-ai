import type { Metadata } from "next";
import "./globals.css";
import { awelier, plexMono, plexSans } from "./fonts";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata: Metadata = {
  title: "ComplAI CORE - Open, reproducible and efficient AI evaluation.",
  description: "Open, calibrated AI evaluations that predict broad benchmark performance from a compact set of tasks.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${awelier.variable} ${plexSans.variable} ${plexMono.variable} scroll-smooth`}>
      <body className="min-h-screen bg-background font-sans antialiased flex flex-col">
        <TooltipProvider>
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
