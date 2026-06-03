import { ImageResponse } from "next/og";
import { loadFaviconFont } from "@/lib/favicon-font";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default async function AppleIcon() {
  const font = await loadFaviconFont();

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#016a71",
          borderRadius: 40,
          color: "#faf8f5",
          fontSize: 62,
          fontFamily: "Noto Serif SC",
          fontWeight: 500,
          letterSpacing: "-0.02em",
        }}
      >
        {"\u89e3\u60d1"}
      </div>
    ),
    {
      ...size,
      fonts: [{ name: "Noto Serif SC", data: font, style: "normal", weight: 500 }],
    },
  );
}
