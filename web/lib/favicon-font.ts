const NOTO_SERIF_SC_MEDIUM =
  "https://cdn.jsdelivr.net/fontsource/fonts/noto-serif-sc@latest/chinese-simplified-500-normal.woff";

let fontCache: ArrayBuffer | null = null;

export async function loadFaviconFont(): Promise<ArrayBuffer> {
  if (fontCache) return fontCache;
  const res = await fetch(NOTO_SERIF_SC_MEDIUM);
  if (!res.ok) {
    throw new Error(`Failed to load favicon font: ${res.status}`);
  }
  fontCache = await res.arrayBuffer();
  return fontCache;
}
