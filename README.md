# vpi-defense

Visual Prompt Injection defense pipeline for multimodal AI accessibility apps.

Sits between user-uploaded images and a multimodal LLM (Gemini API in this
project), targeting VI-user accessibility apps. Detects and sanitizes
injection-style content in images before the model sees them.

## Roadmap

- **Phase 1 — Data (in progress)**
  - M1 Pydantic schemas (image / region / variant) ✅
  - M2 Dataset ingesters (DocVQA, DocLayNet) ✅ (skeleton; verify HF field names on first run)
  - M3 Variant generator (adversarial image synthesis)
  - M4 Label Studio config + adapter
- **Phase 2 — Analysis**
  - M5 Region extraction via Gemini 2.5 Flash (spatial understanding, JSON schema)
  - M6 Risk function (logistic regression) + Integrated Gradients
  - M7 Counterfactual Masking + Gemini generation call
- **Phase 3 — Serving**
  - M8 FastAPI backend
  - M9 React web UI with screen-reader-friendly output

## Requirements

- Python 3.11+
- ~2 GB disk for the initial dataset seed
- Gemini API key (Google AI Studio) for Phase 2 — no local GPU required

## Local setup

```bash
# 1. put the project somewhere sensible, then:
cd vpi_defense

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 3. install this package in editable mode + dev tools
pip install -e ".[dev]"

# 4. run smoke tests (no network required)
pytest tests/ -v
# or run individual test files as scripts:
#   python tests/test_schemas_smoke.py
#   python tests/test_ingest_smoke.py
```

## Fetch initial data (M2)

```bash
mkdir -p data/raw

# DocVQA has no train split; use validation for seeding
python -m vpi_defense.ingest --dataset docvqa    --n 50 --split validation --out data/raw

# DocLayNet has train / validation / test
python -m vpi_defense.ingest --dataset doclaynet --n 50 --split train      --out data/raw
```

Resulting layout:

```
data/raw/
├── docvqa/img_docvqa_val_000000.png
├── docvqa/...
├── doclaynet/img_doclaynet_tra_000000.png
├── doclaynet/...
└── index.jsonl                # one ImageMetadata JSON per line
```

> **On first run**, verify the HF field names for DocLayNet-v1.2 match
> `src/vpi_defense/ingest/doclaynet.py`. Quick check:
>
> ```python
> from datasets import load_dataset
> ds = load_dataset("ds4sd/DocLayNet-v1.2", split="train", streaming=True)
> print(next(iter(ds)).keys())
> ```
>
> If field names differ, adjust `_convert()` in `doclaynet.py`.

## Colab

```python
!git clone <your-repo-url> vpi_defense
%cd vpi_defense
!pip install -e ".[dev]" --quiet

# Optional: persist data to Drive so subsequent runs don't re-download
from google.colab import drive
drive.mount('/content/drive')
!python -m vpi_defense.ingest --dataset docvqa --n 50 \
    --out /content/drive/MyDrive/vpi_defense/data/raw
```

For Phase 2 GPU work, add `[analysis]` extras:
```python
!pip install -e ".[dev,analysis]" --quiet
```

## Vast.ai (optional, only if you switch back to a local VLM)

Not needed for the default Gemini-based pipeline. Use only if you experiment
with an open-weight VLM (PaliGemma, Qwen2.5-VL, etc.) that needs a GPU beyond
Colab's T4 quota. Setup is the same as local, on a rented instance.

## API keys

Phase 2 needs a Gemini API key from [Google AI Studio](https://aistudio.google.com/).
Store it in a `.env` file at the project root (already gitignored):

```
GEMINI_API_KEY=your-key-here
```

Load it in code with `os.environ["GEMINI_API_KEY"]` or via `python-dotenv`.
The `google-genai` client reads this environment variable automatically.

## Project structure

```
vpi_defense/
├── pyproject.toml
├── .gitignore
├── README.md
├── src/vpi_defense/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic models (image / region / variant)
│   └── ingest/               # M2: dataset ingesters
│       ├── __init__.py
│       ├── __main__.py       # CLI entrypoint
│       ├── base.py
│       ├── utils.py
│       ├── docvqa.py
│       └── doclaynet.py
├── tests/
│   ├── test_schemas_smoke.py
│   └── test_ingest_smoke.py
└── data/                     # gitignored; created by ingesters
    └── raw/
```

## Design notes

- `ImageMetadata`, `RegionAnnotation`, `VariantSample` are the single source
  of truth. Everything else (Label Studio exports, ingester outputs,
  downstream analysis) converts to or from these types.
- IDs are prefixed (`img_`, `var_`, `rgn_`) and cross-references are
  validated at load time — see `schemas.DatasetBundle`.
- Variants inherit the split of their base image; do NOT put a `split`
  field on `VariantSample` (it invites leakage).
