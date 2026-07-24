"""Draw risk-colored bounding boxes on an image for the Streamlit demo."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from vpi_defense.pipeline.serve import ScoredRegion


# Risk-tier colors. Matches the semantic tiers used in the app UI.
DANGER_COLOR = (220, 40, 40)      # risk >= 0.7  → red
WARNING_COLOR = (255, 165, 0)     # 0.5 <= risk < 0.7  → orange
SAFE_COLOR = (60, 180, 75)        # risk < 0.5  → green


def risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 0.7:
        return DANGER_COLOR
    if risk >= 0.5:
        return WARNING_COLOR
    return SAFE_COLOR


def draw_regions(
    image: Image.Image,
    regions: list[ScoredRegion],
    show_labels: bool = True,
    line_width: int = 3,
    show_only_flagged: bool = False,
) -> Image.Image:
    """Return a copy of the image with each region's bbox drawn.

    Args:
        image            : source PIL image
        regions          : ScoredRegion objects from run_pipeline
        show_labels      : draw the risk score above each box
        line_width       : bbox line thickness (pixels)
        show_only_flagged: if True, only draw regions with risk >= 0.5
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    font_size = max(14, min(w, h) // 60)
    try:
        # Prefer a CJK-capable font so Korean OCR text renders correctly.
        from vpi_defense.variants.rendering import load_font
        font = load_font(size=font_size)
    except Exception:
        font = ImageFont.load_default()

    for region in regions:
        if show_only_flagged and region.risk_prob < 0.5:
            continue

        color = risk_color(region.risk_prob)
        x0 = int(region.bbox.x_min * w)
        y0 = int(region.bbox.y_min * h)
        x1 = int(region.bbox.x_max * w)
        y1 = int(region.bbox.y_max * h)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        if show_labels:
            label = f"{region.risk_prob:.2f}"
            # Filled background behind the label for readability
            label_y = max(0, y0 - font_size - 4)
            text_bbox = draw.textbbox((x0, label_y), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x0 + 2, label_y), label, fill=(255, 255, 255), font=font)

    return out


def summarize_regions(regions: list[ScoredRegion], top_k: int = 5) -> list[dict]:
    """Top-K regions by risk, as plain dicts for st.dataframe / st.table."""
    top = sorted(regions, key=lambda r: -r.risk_prob)[:top_k]
    return [
        {
            "risk": round(r.risk_prob, 3),
            "type": r.region_type,
            "ocr_text": (r.ocr_text or "")[:80],
        }
        for r in top
    ]
