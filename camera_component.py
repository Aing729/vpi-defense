"""Streamlit Custom Component: 전면/후면 전환 가능한 카메라.

사용:
    from camera_component import camera_input
    captured = camera_input(key="cam")
    if captured:  # base64 data URL
        ...
"""
from pathlib import Path
import streamlit.components.v1 as components

_FRONTEND_PATH = Path(__file__).parent / "camera_frontend"

_camera_component = components.declare_component(
    "vpi_camera",
    path=str(_FRONTEND_PATH),
)


def camera_input(key=None):
    """카메라 UI 표시 + 캡처 결과 반환.

    Returns:
        str | None: base64 data URL ("data:image/jpeg;base64,...") 또는 None
    """
    return _camera_component(key=key, default=None)
