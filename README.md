# OCR Evaluation Framework

A unified Python framework to evaluate open-source OCR models on a standardized test dataset.
Captures accuracy, latency, throughput, cost, and structured extraction quality.

## Quick Start

### macOS / Linux
```bash
# 1. Clone and setup
git clone <your-repo-url>
cd ocr-eval-framework
chmod +x setup_env.sh
./setup_env.sh

# 2. Activate environment
source venv/bin/activate

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 4. Test dataset is already included — verify it's there
ls test-dataset/

# 5. Run evaluations
python run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png
python run_model.py --model mistral_ocr
python run_batch.py

# 6. Generate report
python evaluate.py --results-dir results/latest
```

### Windows
```powershell
# 1. Clone and setup
git clone <your-repo-url>
cd ocr-eval-framework

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# 4. Install Tesseract (download from https://github.com/UB-Mannheim/tesseract/wiki)
#    Add Tesseract install path to your system PATH

# 5. Configure API keys
copy .env.example .env
# Edit .env with your API keys

# 6. Test dataset is already included — verify it's there
dir test-dataset

# 7. Run evaluations
python run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png
python run_model.py --model mistral_ocr
python run_batch.py

# 8. Generate report
python evaluate.py --results-dir results/latest
```

## Test Dataset

The curated test dataset (110-160 documents) is committed to the repo under `test-dataset/`. **No need to run the download script** — just clone the repo and you're ready to go.

The dataset covers:
- Printed English (invoices, contracts, reports)
- Complex tables & layouts (financial, forms, nested)
- Handwritten (English + Indian languages)
- Indian languages (Hindi, Telugu, Tamil, Bengali)
- European languages (Spanish, French, German)
- Low quality scans (faded, skewed, noisy)
- Mixed content (images, equations, receipts)
- Organization internal documents

Ground truth files are in `test-dataset/ground_truth/` and are automatically matched by the framework during evaluation.

To add more documents later, you can still run:
```bash
python download_dataset.py --output-dir ./test-dataset --samples 5
```

## Models Supported (10)

| # | Model | Type | GPU Needed? | Setup |
|---|-------|------|-------------|-------|
| 1 | Tesseract | Open Source | No (CPU) | System install |
| 2 | PaddleOCR | Open Source | Optional | pip install |
| 3 | Docling/SmolDocling | Open Source | Optional | pip install |
| 4 | Surya OCR | Open Source | Optional | pip install |
| 5 | Mistral OCR | API | No | API key |
| 6 | Sarvam Vision OCR | API | No | API key |
| 7 | DeepSeek OCR | Open Source | Yes (16GB+) | HuggingFace model |
| 8 | olmOCR | Open Source | Yes (16GB+) | HuggingFace model |
| 9 | Qwen2.5-VL | Open Source | Yes (16GB+) | HuggingFace model |
| 10 | GOT-OCR 2.0 | Open Source | Yes (8-16GB) | HuggingFace model |

## Prerequisites

### HuggingFace Setup (Optional — only if adding more datasets)

The test dataset is already committed to the repo. HuggingFace login is only needed if you want to download additional documents using the download script.

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens
3. Click **"New token"** → name it anything → select **"Read"** access → create
4. In your terminal:
```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your token when prompted
```
5. Some datasets are **"gated"** — visit these pages and click **"Accept"** if prompted:
   - https://huggingface.co/datasets/getomni/ocr-benchmark
   - https://huggingface.co/datasets/unstructured-io/SCORE-Bench
   - https://huggingface.co/datasets/aharley/rvl_cdip

### API Keys (Mistral OCR, Sarvam OCR)

API keys are stored in `.env` (gitignored) and automatically loaded at runtime.
Copy `.env.example` to `.env` and fill in your credentials:
```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Non-secret settings (model IDs, regions, languages) remain in `configs/config.yaml`.

## First Run Checklist (Verified ✅)

After setup, validate the pipeline works with this exact command:
```bash
python run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png
```

**Expected output:** ~495 chars extracted, CER ~0.53, latency ~4000ms

Then test a cloud model (requires Mistral API key in `.env`):
```bash
python run_single.py --model mistral_ocr --input test-dataset/02_complex_tables/forms/0012199830.png
```

**Expected output:** ~517 chars extracted, CER ~0.33, latency ~1700ms

If both commands produce results with metrics, the framework is working correctly.

## Usage
```bash
# List all available models
python run_batch.py --list

# Run one model on one document
python run_single.py --model tesseract --input path/to/document.jpg

# Run one model on the entire dataset
python run_model.py --model mistral_ocr

# Run all models on all documents (full evaluation)
python run_batch.py

# Run specific models only
python run_batch.py --models tesseract paddleocr mistral_ocr surya

# Generate comparison report + CSV export
python evaluate.py --results-dir results/latest --export-csv
```

## Platform Notes

### Apple Silicon (macOS)

- Models marked "Optional" GPU work on M-series chips via MPS
- 32GB+ unified memory can handle 7B-9B parameter models
- Set `device: mps` in config for Apple Silicon
- For 30B+ models, use cloud GPU (Colab Pro or equivalent)

### Windows

- Python 3.10+ required — download from [python.org](https://www.python.org/downloads/)
- Tesseract must be installed separately — download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- PaddleOCR: install `paddlepaddle` (CPU) or `paddlepaddle-gpu` (CUDA) — see [PaddlePaddle docs](https://www.paddlepaddle.org.cn/en/install/quick)
- CUDA GPU users: install PyTorch with CUDA support — `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`

### Linux

- Install Tesseract via package manager: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-ben`
- CUDA GPU users: install PyTorch with CUDA — `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- CPU-only: cloud API models work fine; open-source VLMs will be slow

## Known Issues

- **Tesseract not found error**: Tesseract is a system-level install (C++ binary), not a pip package. The `pytesseract` pip package is just a Python wrapper. Install the engine separately:
  - macOS: `brew install tesseract tesseract-lang`
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel tesseract-ocr-tam tesseract-ocr-ben`
  - Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Mistral SDK import**: Use `from mistralai.client import Mistral` (not `from mistralai import Mistral`) — the import path changed in SDK v2
- **Tesseract multi-lang garbling**: Setting `lang: eng+hin+tel+tam+ben` can cause English text to be misread as Indic scripts. Use `lang: eng` for English-only documents in `configs/config.yaml`
- **HuggingFace 401 errors**: Run `huggingface-cli login` and accept gated dataset terms (see Prerequisites section above)
- **IndicPhotoOCR download fails**: The download script may fail for some image URLs. Clone the repo manually instead:
```bash
  git clone https://github.com/Bhashini-IITJ/IndicPhotoOCR.git /tmp/IndicPhotoOCR
  cp /tmp/IndicPhotoOCR/test_images/*.jpg test-dataset/04_indian_languages/hindi/
  rm -rf /tmp/IndicPhotoOCR
```
- **PaddleOCR on Apple Silicon**: If you get GPU errors, set `use_gpu: false` in `configs/config.yaml` under the `paddleocr` section
- **Large VLM models (DeepSeek, Qwen, olmOCR)**: First run downloads several GB of model weights. Ensure sufficient disk space and a stable internet connection

## Documentation

| File | Description |
|------|-------------|
| [docs/OCR_Whitepaper_Plan_v2.docx](docs/OCR_Whitepaper_Plan_v2.docx) | Whitepaper plan — research methodology, evaluation criteria, and paper outline |
| [docs/OCR_Test_Dataset_Tracker_v2.xlsx](docs/OCR_Test_Dataset_Tracker_v2.xlsx) | Test dataset tracker — document inventory, model registry, metrics guide, team assignments, references |

## Project Structure
```
ocr-eval-framework/
├── setup_env.sh              # Environment setup script (macOS/Linux)
├── download_dataset.py       # Download additional test documents
├── requirements.txt          # Python dependencies
├── .env.example              # API key template (copy to .env)
├── docs/
│   ├── OCR_Whitepaper_Plan_v2.docx       # Research plan & paper outline
│   └── OCR_Test_Dataset_Tracker_v2.xlsx  # Dataset tracker (9 sheets)
├── configs/
│   └── config.yaml           # Model settings (non-secret config)
├── models/
│   ├── __init__.py           # Model registry
│   ├── base.py               # Base interface + OCRResult dataclass
│   ├── tesseract_model.py    # Tesseract (baseline)
│   ├── paddleocr_model.py    # PaddleOCR
│   ├── docling_model.py      # Docling / SmolDocling
│   ├── surya_model.py        # Surya OCR
│   ├── mistral_ocr.py        # Mistral OCR
│   ├── deepseek_ocr.py       # DeepSeek OCR
│   ├── olmocr_model.py       # olmOCR
│   ├── qwen_vl.py            # Qwen2.5-VL
│   ├── got_ocr.py            # GOT-OCR 2.0
│   └── sarvam_ocr.py         # Sarvam Vision OCR
├── utils/
│   ├── metrics.py            # CER, WER, F1, table accuracy
│   ├── runner.py             # Batch execution engine
│   ├── report.py             # HTML report generator
│   └── helpers.py            # Shared utilities
├── run_single.py             # One model + one document
├── run_model.py              # One model + all documents
├── run_batch.py              # All models + all documents
├── evaluate.py               # Compute metrics + generate report
├── test-dataset/             # Curated test documents (committed to repo)
└── results/                  # Evaluation results (gitignored)
```