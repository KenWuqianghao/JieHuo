#!/usr/bin/env python3
"""Generate a Twitter-ready JieHuo metric card."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
OUT = ASSETS_DIR / "jiehuo-twitter-card.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 1)


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "#f7f5ef")
    draw = ImageDraw.Draw(img)

    # Structured, presentation-like background.
    draw.rectangle((0, 0, width, 130), fill="#111827")
    draw.rectangle((0, 130, width, height), fill="#f7f5ef")
    draw.rectangle((0, height - 110, width, height), fill="#e7efe9")

    draw.text((72, 42), "JieHuo", fill="#ffffff", font=font(54, True))
    draw.text((302, 56), "12-language Google vs Perplexity router", fill="#d1d5db", font=font(30))

    draw.text((72, 180), "Browser-runnable search routing", fill="#111827", font=font(64, True))
    draw.text(
        (72, 260),
        "GPT-4.1-mini teacher-distilled -> multilingual-e5-small -> calibrated INT8 ONNX",
        fill="#374151",
        font=font(31),
    )

    cards = [
        ("Balanced gold", "0.883", "macro F1", "full coverage"),
        ("Auto-route", "0.911", "macro F1", "88.8% coverage"),
        ("High confidence", "0.962", "macro F1", "56.7% coverage"),
    ]
    x = 72
    for title, value, metric, caption in cards:
        rounded(draw, (x, 360, x + 460, 640), 20, "#ffffff", "#d1d5db")
        draw.text((x + 34, 394), title, fill="#374151", font=font(28, True))
        draw.text((x + 34, 452), value, fill="#0f766e", font=font(84, True))
        draw.text((x + 260, 490), metric, fill="#111827", font=font(32, True))
        draw.text((x + 34, 568), caption, fill="#4b5563", font=font(30))
        x += 500

    draw.text((72, 704), "Runs locally in the browser with transformers.js", fill="#111827", font=font(34, True))
    draw.text((72, 752), "Model: KenWu/multilingual-query-router on Hugging Face", fill="#374151", font=font(28))
    draw.text((72, 826), "github.com/KenWuqianghao/JieHuo", fill="#111827", font=font(30, True))
    draw.text((1060, 826), "HF: KenWu/multilingual-router", fill="#111827", font=font(26, True))

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
