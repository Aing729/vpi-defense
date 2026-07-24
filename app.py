"""VPI Defense — 데모 시연용 Streamlit 앱.

4섹션 구조:
  1. 이미지 업로드 + 실행
  2. Before / After 이미지 비교
  3. 안전 응답 (텍스트 + TTS)
  4. 안전 브리핑

Run locally:
    streamlit run app.py

Required environment / secrets:
    GEMINI_API_KEY : Google AI Studio API key
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vpi_defense.analysis import GeminiVLMClient
from vpi_defense.features import try_load_default_embedder
from vpi_defense.pipeline import apply_masks, run_pipeline
from vpi_defense.risk import RiskModel
from vpi_defense.webapp import draw_regions, synthesize_speech_bytes


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VPI Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",  # 시연 시 사이드바 접힘 — 화면 넓게
)


def _load_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "risk_v1_with_embedder.pkl"


@st.cache_resource(show_spinner="위험도 모델 로드 중...")
def load_risk_model(path: Path):
    return RiskModel.load(path)


@st.cache_resource(show_spinner="언어 모델 로드 중 (첫 실행 시 120MB 다운로드)...")
def load_embedder():
    return try_load_default_embedder()


@st.cache_resource
def get_vlm_client():
    api_key = _load_api_key()
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    return GeminiVLMClient()


# ---------------------------------------------------------------------------
# Sidebar (접혀 있음, 시연 중 필요 시 열기)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ 설정")
    risk_threshold = st.slider("위험도 임계값", 0.0, 1.0, 0.5, 0.05)
    similarity_threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.85, 0.05)
    mask_mode = st.selectbox(
        "마스킹 방식", ["blur", "solid"],
        format_func=lambda x: {"blur": "가우시안 블러", "solid": "단색"}[x],
    )
    enable_tts = st.checkbox("TTS 자동 재생", value=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ VPI Defense")
st.markdown(
    "**시각장애인용 이미지 안전 분석 시스템**  \n"
    "이미지 속 숨겨진 명령·링크로부터 사용자를 보호합니다."
)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

api_key = _load_api_key()
if not api_key:
    st.error("🔑 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

if not _DEFAULT_MODEL_PATH.exists():
    st.error(f"📁 모델 파일 없음: `{_DEFAULT_MODEL_PATH}`")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1: 이미지 업로드 + 실행
# ═════════════════════════════════════════════════════════════════════════

st.divider()

uploaded_file = st.file_uploader(
    "🖼️ 분석할 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("이미지를 업로드하면 분석이 시작됩니다.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col_preview, col_action = st.columns([2, 1])
with col_preview:
    st.image(image, caption="업로드된 이미지", use_column_width=True)
with col_action:
    st.markdown("### 실행 준비")
    st.markdown(f"- 크기: {image.width} × {image.height} px")
    st.markdown(f"- 파일: `{uploaded_file.name}`")
    st.markdown("")
    run = st.button("🚀 안전 분석 시작", type="primary", use_container_width=True)

if not run:
    st.stop()


# ---------------------------------------------------------------------------
# Load resources + run pipeline
# ---------------------------------------------------------------------------

risk_model = load_risk_model(_DEFAULT_MODEL_PATH)
embedder = load_embedder()
vlm = get_vlm_client()

if vlm is None:
    st.error("VLM 클라이언트 초기화 실패")
    st.stop()

with st.spinner("🔍 파이프라인 실행 중... (10-30초 소요)"):
    try:
        result = run_pipeline(
            image=image, vlm_client=vlm, risk_model=risk_model, embedder=embedder,
            risk_threshold=risk_threshold,
            similarity_threshold=similarity_threshold,
            mask_mode=mask_mode,
        )
    except Exception as e:
        st.error(f"파이프라인 실행 실패: {e}")
        st.exception(e)
        st.stop()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2: Before / After 비교
# ═════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🔍 위험 영역 분석")

if not result.masks:
    st.success(
        f"✅ 위험 영역이 감지되지 않았습니다. 원본 이미지를 그대로 사용합니다."
    )
    vis_all = draw_regions(image, result.regions, show_labels=True, line_width=3)
    st.image(vis_all, caption="검출된 영역 (초록 = 안전)", use_column_width=True)
else:
    st.markdown(
        f"위험도 **{risk_threshold:.2f} 이상**인 영역 **{len(result.masks)}개**를 감지하여 마스킹했습니다."
    )
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("#### Before (원본 + 위험 영역 표시)")
        vis_all = draw_regions(image, result.regions, show_labels=True, line_width=3)
        st.image(vis_all, use_column_width=True)
    with col_after:
        st.markdown("#### After (위험 영역 마스킹됨)")
        masked_image = apply_masks(image, result.masks, mode=mask_mode)
        st.image(masked_image, use_column_width=True)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3: 안전 응답 + TTS
# ═════════════════════════════════════════════════════════════════════════

st.divider()
st.header("✅ 안전 응답")

st.markdown(
    f"<div style='font-size: 1.15em; line-height: 1.7em; padding: 1.5em; "
    f"background-color: #f0f4f8; border-left: 6px solid #4a90e2; "
    f"border-radius: 6px;'>{result.safe_response}</div>",
    unsafe_allow_html=True,
)

if enable_tts:
    with st.spinner("음성 생성 중..."):
        audio_bytes = synthesize_speech_bytes(result.safe_response)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 4: 안전 브리핑
# ═════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🔔 안전 브리핑")


def _get_briefing_content(result) -> tuple[str, str]:
    """상황별 브리핑 메시지 생성. 반환: (severity, message).
    severity ∈ {'danger', 'info', 'success'}
    """
    if not result.masks:
        # 정상 이미지
        return "success", (
            "**이 이미지는 안전합니다.**\n\n"
            "이미지에서 위험한 명령·링크·QR 코드 등이 감지되지 않았습니다. "
            "원본 응답을 그대로 사용자에게 전달합니다."
        )

    if result.comparison and result.comparison.changed:
        # Injection 감지 + CFM 개입 성공
        # 어떤 종류의 위험 요소였는지 상위 위험 영역에서 파악
        top_risky = sorted(result.regions, key=lambda r: -r.risk_prob)[:3]
        risky_types = list({r.region_type for r in top_risky if r.risk_prob >= 0.5})
        risky_texts = [
            (r.ocr_text or "")[:60] for r in top_risky
            if r.risk_prob >= 0.5 and r.ocr_text
        ][:2]

        elements = []
        if any('background' in t or 'small' in t for t in risky_types):
            elements.append("숨겨진 텍스트")
        if any('button' in t for t in risky_types):
            elements.append("의심스러운 링크·버튼")
        if not elements:
            elements = ["명령형 문구"]

        detected = ", ".join(elements)
        text_hint = ""
        if risky_texts:
            text_hint = f"\n- **감지된 텍스트 예시**: {' / '.join(f'"{t}..."' for t in risky_texts)}\n"

        return "danger", (
            f"⚠️ **이 이미지에 위험 요소가 포함되어 있어 자동으로 제거했습니다.**\n\n"
            f"- **감지된 위험 요소**: {detected}\n"
            f"- **처리 결과**: {len(result.masks)}개 영역 마스킹, 안전한 응답으로 재구성\n"
            f"{text_hint}"
            f"- **주의사항**: 링크 접속, QR 스캔, 개인정보 입력을 삼가세요."
        )

    # 마스킹은 있었지만 CFM이 원본 유지 결정 (false positive 회복)
    return "info", (
        f"ℹ️ **위험 가능성이 있는 영역 {len(result.masks)}개를 검토했지만, "
        f"실제 위협은 아닌 것으로 판단되었습니다.**\n\n"
        f"- **처리 결과**: 마스킹 후 응답이 원본과 거의 동일하여 원본 응답을 유지합니다.\n"
        f"- **의미**: 이 이미지는 안전합니다. 정보 손실 없이 원본 그대로 전달됩니다."
    )


severity, message = _get_briefing_content(result)

if severity == "danger":
    st.error(message)
elif severity == "info":
    st.info(message)
else:
    st.success(message)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "🎓 VPI Defense — Visual Prompt Injection Defense Pipeline  ·  "
    "위험도 함수: L2 Logistic Regression (10 features, AUC=0.940)  ·  "
    "CFM: Counterfactual Feature Masking with semantic similarity check"
)
