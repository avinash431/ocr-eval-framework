---
name: add-ocr-model
description: Add a new OCR model wrapper to the evaluation framework. Use when the user wants to integrate a new OCR engine, add a new cloud API model, wrap a new vision language model, or create a new model adapter. Triggers on phrases like "add model", "integrate OCR", "new model wrapper", "support for X model", or any mention of adding an OCR provider.
---

# Add OCR Model

Scaffold and integrate a new OCR model into the evaluation framework following the established registry pattern.

## Before You Start

1. Read `models/base.py` to understand `BaseOCRModel` and `OCRResult`
2. Read `models/__init__.py` to understand the `@register_model` decorator and auto-discovery
3. Ask the user:
   - What is the model name and provider?
   - Is it a cloud API, open-source local model, or vision language model (VLM)?
   - What authentication/credentials does it need?
   - Does it return structured data (tables, key-value pairs)?

## Model Types and Patterns

There are three model categories — follow the matching pattern:

### Cloud API Models (e.g., Azure, Google, Mistral)
- Authentication via `.env` environment variables
- Cost estimation in `estimate_cost()`
- Network calls with proper error handling
- Reference: `models/azure_adi.py`, `models/mistral_ocr.py`

### Open-Source Local Models (e.g., Tesseract, PaddleOCR)
- Config-driven parameters (language, modes)
- Optional GPU support via `get_device(config)`
- System dependencies documented
- Reference: `models/tesseract_model.py`, `models/paddleocr_model.py`

### Vision Language Models (e.g., Qwen, DeepSeek)
- HuggingFace Transformers loading
- GPU/MPS device placement via `get_device(config)`
- Image preprocessing with PIL
- Chat/generation interface for OCR prompts
- Reference: `models/qwen_vl.py`, `models/deepseek_ocr.py`

## Step-by-Step

### 1. Create the model wrapper

Create `models/<model_name>.py` using this template:

```python
"""<Model Display Name> wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class ModelClassName(BaseOCRModel):
    @property
    def name(self): return "<model_name>"
    @property
    def display_name(self): return "<Model Display Name>"
    @property
    def model_type(self): return "cloud_api"  # or "open_source"

    def setup(self):
        # Import dependencies, authenticate, load weights
        cfg = self.config.get("<config_section>", {})
        # ... initialization ...
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        # Core OCR logic — read image, call model, extract text
        raw_text = ""
        return OCRResult(
            raw_text=raw_text,
            metadata={},
        )

    def estimate_cost(self, num_pages: int) -> float:
        return 0.0  # Cloud APIs: calculate per-page cost
```

### 2. Add configuration

Add a section to `configs/config.yaml` with non-secret settings:

```yaml
<model_name>:
  model: "model-identifier"
  # Other non-secret parameters
```

### 3. Add environment variables (for cloud APIs)

If the model needs API keys:

1. Add env vars to `.env.example` and `.env`:
   ```
   MODEL_API_KEY=""
   ```

2. Add the mapping in `utils/helpers.py` under `_ENV_TO_CONFIG`:
   ```python
   "MODEL_API_KEY": ("<config_section>", "api_key"),
   ```

### 4. Add dependencies

- Add Python packages to `requirements.txt`
- For system dependencies (like Tesseract), document install instructions in the model file docstring and README

### 5. Verify

```bash
source venv/bin/activate
python -c "from models import list_models; print(list_models())"
python run_single.py --model <model_name> --input <test_image>
```

## Important Rules

- The `@register_model` decorator auto-registers the model — no manual registration needed
- Always set `self._is_setup = True` at the end of `setup()`
- Return `OCRResult` with at minimum `raw_text` — all other fields are optional
- Use lazy imports inside `setup()` for heavy dependencies so the framework loads fast
- Handle both image files (.jpg, .png) and PDFs where the model supports it
- Normalize confidence scores to 0.0–1.0 range
- Include cost estimation for cloud APIs (check provider pricing docs)
