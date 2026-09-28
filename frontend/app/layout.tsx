import type { Metadata } from "next";
import { Geist_Mono, Inter } from "next/font/google";
import "./globals.css";

/**
 * Inter for everything that is read, Geist Mono for the one thing that is
 * compared — the `node 12345` ids in the stop list, where tabular figures
 * matter more than elegance.
 *
 * `next/font` self-hosts these at build time, so there is no request to Google
 * from the browser and no layout shift while a webfont swaps in.
 */
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Q-Gati — Vehicle routing for Delhi",
  description:
    "Multi-algorithm vehicle routing on the real Delhi road network, with live results from the Q-Gati FastAPI backend.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
