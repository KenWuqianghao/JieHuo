import type { Metadata } from "next";
import { DM_Sans, Newsreader } from "next/font/google";
import "./globals.css";

const sans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});

const serif = Newsreader({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600"],
});

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "https://jiehuo.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "解惑 JieHuo — Query Router",
  description:
    "12-language browser router that decides Google vs Perplexity. Teacher-distilled, calibrated INT8 ONNX — runs locally in your browser.",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: siteUrl,
    siteName: "JieHuo",
    title: "解惑 JieHuo — Google vs Perplexity Router",
    description:
      "Multilingual query router with 0.883 macro F1 on balanced gold. Auto-route high-confidence queries; ambiguous ones stay with you.",
    images: [
      {
        url: "/opengraph-image.png",
        width: 1200,
        height: 630,
        alt: "JieHuo — 12-language Google vs Perplexity query router",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "解惑 JieHuo — Google vs Perplexity Router",
    description:
      "12-language browser router. 0.883 macro F1 · runs locally with INT8 ONNX. Try it free.",
    images: ["/opengraph-image.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
