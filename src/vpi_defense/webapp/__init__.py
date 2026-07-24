"""Streamlit web app for the VPI-defense pipeline."""
from __future__ import annotations

from .tts import synthesize_speech_bytes
from .visualize import draw_regions, risk_color, summarize_regions

__all__ = [
    "draw_regions",
    "risk_color",
    "summarize_regions",
    "synthesize_speech_bytes",
]
