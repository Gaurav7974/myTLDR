import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono, Silkscreen } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jbmono",
  subsets: ["latin"],
});

const silkscreen = Silkscreen({
  weight: ["400", "700"],
  variable: "--font-pixel",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "myTLDR / Understand any research paper",
  description: "Paste a paper link or drop a PDF and get a plain-language explanation at your depth, plus a verified architecture diagram.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} ${silkscreen.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
