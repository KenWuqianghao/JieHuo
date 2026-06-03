#!/usr/bin/env python3
"""Generate Open Graph / Twitter card image for JieHuo."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
OUT = ASSETS_DIR / "jiehuo-twitter-card.png"
WEB_OG = ROOT / "web" / "app" / "opengraph-image.png"

# Site palette (web/app/globals.css)
CANVAS = "#faf8f5"
INK = "#271a00"
INK_SECONDARY = "#72706b"
TEAL = "#016a71"
TEAL_SOFT = "#ddf6f8"
GOOGLE = "#1a73e8"
WHITE = "#ffffff"


def font(size: int, bold: bool = False, cn: bool = False) -> ImageFont.FreeTypeFont:
    if cn:
        candidates = [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: str | None = None,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 0)


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    width, height = 1200, 630  # 1.91:1 — ideal for X / Open Graph
    img = Image.new("RGB", (width, height), CANVAS)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for y in range(260):
        t = 1 - y / 260
        ov_draw.rectangle((0, y, width, y + 1), fill=(1, 106, 113, int(16 * t)))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Header bar
    rounded(draw, (48, 48, width - 48, 148), 20, TEAL)
    draw.text((80, 72), "\u89e3\u60d1", fill=WHITE, font=font(52, cn=True))
    draw.text((210, 78), "JieHuo", fill=WHITE, font=font(44, bold=True))
    draw.text((210, 118), "Multilingual Google vs Perplexity router", fill=TEAL_SOFT, font=font(22))

    draw.text((80, 196), "Route every query to the right search engine", fill=INK, font=font(46, bold=True))
    draw.text(
        (80, 258),
        "Runs in your browser · GPT-4.1-mini distilled · INT8 ONNX · 12 languages",
        fill=INK_SECONDARY,
        font=font(24),
    )

    cards = [
        ("Balanced gold", "0.883", "macro F1 · full coverage"),
        ("Auto-route", "0.911", "macro F1 · 89% coverage"),
        ("High confidence", "0.962", "macro F1 · 57% coverage"),
    ]
    x = 80
    for title, value, caption in cards:
        rounded(draw, (x, 310, x + 340, 500), 16, WHITE, "#e8e4dc")
        draw.rectangle((x + 24, 334, x + 52, 342), fill=TEAL)
        draw.text((x + 24, 352), title, fill=INK_SECONDARY, font=font(20, bold=True))
        draw.text((x + 24, 392), value, fill=TEAL, font=font(58, bold=True))
        draw.text((x + 24, 458), caption, fill=INK_SECONDARY, font=font(20))
        x += 360

    # Route chips
    rounded(draw, (80, 530, 310, 582), 999, "#e8f1fc")
    draw.text((108, 544), "Google", fill=GOOGLE, font=font(22, bold=True))
    rounded(draw, (330, 530, 560, 582), 999, TEAL_SOFT)
    draw.text((352, 544), "Perplexity", fill=TEAL, font=font(22, bold=True))
    draw.text((590, 544), "jiehuo.vercel.app", fill=INK, font=font(22, bold=True))

    img.save(OUT, quality=92)
    WEB_OG.parent.mkdir(parents=True, exist_ok=True)
    img.save(WEB_OG, quality=92)
    print(f"Wrote {OUT}")
    print(f"Wrote {WEB_OG}")


if __name__ == "__main__":
    main()
