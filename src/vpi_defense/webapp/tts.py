"""Text-to-speech synthesis for the accessibility feature.

Uses Google's gTTS (free, no API key needed, network required). Auto-detects
Korean vs English based on character range; falls back to English on ambiguity.
"""
from __future__ import annotations

import io
import re


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def detect_language(text: str) -> str:
    """Return 'ko' if the text contains any Hangul syllables, else 'en'.

    A more sophisticated approach would count character frequencies, but
    Korean and English are our only supported languages and mixed-language
    text usually reads acceptably in Korean voice.
    """
    if _HANGUL_RE.search(text):
        return "ko"
    return "en"


def synthesize_speech_bytes(text: str, lang: str | None = None) -> bytes:
    """Convert text to MP3 audio bytes for streaming to the browser.

    Args:
        text : the text to speak
        lang : 'ko', 'en', or None (auto-detect)

    Returns:
        MP3 audio as bytes, ready to pass to `st.audio(audio_bytes)`.
        Returns empty bytes on any failure so the UI can degrade gracefully.
    """
    if not text or not text.strip():
        return b""

    if lang is None:
        lang = detect_language(text)

    try:
        from gtts import gTTS
    except ImportError:
        # gTTS not installed → silently skip audio. UI will still show text.
        return b""

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        # Network errors, quota, etc — again, degrade gracefully
        return b""
