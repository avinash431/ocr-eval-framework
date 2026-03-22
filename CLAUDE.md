# OCR Evaluation Framework

## Overview
Unified Python framework evaluating 14 OCR models on standardized test datasets. Captures accuracy (CER, WER, F1, BLEU), latency, throughput, cost, and structured extraction quality.

## Key Commands
```bash
source venv/bin/activate              # Activate environment
python run_single.py --model X --input Y  # Test one model on one doc
python run_model.py --model X          # One model, full dataset
python run_batch.py                    # All models, full dataset
python evaluate.py --results-dir Z     # Generate HTML report
python download_dataset.py             # Download test documents
```

## Architecture
- **models/**: Each file is a self-registering OCR wrapper via `@register_model`
- **utils/**: Runner, metrics, report generation, helpers
- **configs/**: YAML config for non-secret settings; secrets in `.env`
- New models are added by creating a file in `models/` — no manual registration needed

## Conventions
- API keys go in `.env`, never in YAML or code
- All models inherit `BaseOCRModel` and implement `_ocr_impl()`
- Results are saved to timestamped dirs under `results/`
- Ground truth files use `<stem>_gt.txt` naming
