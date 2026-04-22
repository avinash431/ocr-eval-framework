# Reproducibility Guide

This document provides all information needed to reproduce the benchmark results
reported in the OCR evaluation whitepaper.

## Environment

| Component | Version |
|-----------|---------|
| OS | macOS 15.5 (Darwin 24.5.0) |
| Architecture | Apple Silicon (arm64, M4 Air 16GB) |
| Python | 3.12.9 |
| PyTorch | 2.10.0 |
| Transformers | 4.57.6 |
| PaddlePaddle | 3.3.0 |
| PaddleOCR | 3.4.0 |
| Surya OCR | 0.17.1 |
| Docling | 2.81.0 |
| Matplotlib | 3.10.8 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd ocr-eval-framework
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API keys (required for Mistral OCR and Sarvam OCR)
cp .env.example .env
# Edit .env with your API keys
```

## Dataset

The test dataset contains 101 documents across 4 active categories:

| Category | Documents | Description |
|----------|-----------|-------------|
| 02_complex_tables/financial | 15 | Financial statements, balance sheets |
| 02_complex_tables/forms | 5 | FUNSD form documents (human-annotated GT) |
| 02_complex_tables/multi_column | 15 | Multi-column layouts |
| 03_handwritten/hindi_devanagari | 30 | DHCD 32×32 character crops |
| 04_indian_languages/hindi | 1 | Hindi document |
| 06_mixed_content/equations_formulas | 15 | Mathematical equations and formulas |
| 06_mixed_content/receipts | 10 | Receipt images |

```bash
python tools/download_dataset.py  # Download test documents
```

## Ground Truth

Ground truth comes from two sources:

1. **Human-annotated** (15 docs): FUNSD forms (5) + receipts (10)
   - Location: `test-dataset/ground_truth/`
   - Format: `<stem>_gt.txt`

2. **Model-consensus** (30 docs): Generated via cross-model agreement
   - Primary source: Mistral OCR output
   - Cross-validated against 3+ other models using token F1
   - Minimum consensus threshold: F1 ≥ 0.3
   - Categories: financial (9), multi_column (11), equations (10)
   - **Circularity caveat**: Mistral OCR scores are artificially inflated on
     consensus GT categories since it was the primary source

```bash
# To regenerate consensus GT (will not overwrite existing human GT):
python tools/generate_ground_truth.py --dry-run          # Preview
python tools/generate_ground_truth.py                     # Generate
```

## Running Evaluations

### Single document
```bash
python cli/run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png
```

### Full dataset (single model)
```bash
python cli/run_model.py --model tesseract
python cli/run_model.py --model mistral_ocr
python cli/run_model.py --model surya
python cli/run_model.py --model docling
python cli/run_model.py --model paddleocr
python cli/run_model.py --model sarvam_ocr
python cli/run_model.py --model got_ocr
```

### Full dataset (all models)
```bash
python cli/run_batch.py
```

### Recompute metrics against expanded GT
```bash
python tools/recompute_metrics.py
```

### Generate charts
```bash
python tools/generate_charts.py
```

## Metrics Definitions

All metrics are computed after text normalization (Unicode NFKC, markdown/HTML
stripping, whitespace collapse). Implementation: `utils/metrics.py`.

| Metric | Definition | Range | Better |
|--------|-----------|-------|--------|
| CER | Character Error Rate = edit_distance(pred, gt) / len(gt) | 0–∞ | Lower |
| WER | Word Error Rate = edit_distance(pred_words, gt_words) / len(gt_words) | 0–∞ | Lower |
| Token F1 | Counter-based bag-of-words F1 (with multiplicity) | 0–1 | Higher |
| Precision | TP / (TP + FP) at token level | 0–1 | Higher |
| Recall | TP / (TP + FN) at token level | 0–1 | Higher |
| Word Accuracy | max(0, 1 - WER) | 0–1 | Higher |
| Edit Distance | Normalized Levenshtein distance | 0–1 | Lower |
| S/I/D | Substitution, Insertion, Deletion counts from Levenshtein alignment | counts | Lower |

**Note on F1**: Uses `collections.Counter` (bag-of-words with multiplicity),
not set-based. This means repeated words are counted correctly — a word
appearing 3× in prediction and 2× in ground truth contributes 2 true positives.

## Statistical Tests

Pairwise comparisons use:
- **Wilcoxon signed-rank test** (non-parametric, paired)
- **Cohen's d** effect sizes
- Significance threshold: p < 0.05

```bash
# Statistical tests are computed as part of recompute_metrics.py
# Results saved to: results/expanded_gt_metrics/statistical_tests.json
```

## Configuration

Non-secret model settings: `configs/config.yaml`
API keys: `.env` (never committed)
Timeout per document: 120 seconds (configurable in config.yaml)

## Results Artifacts

All results are saved to timestamped directories under `results/`:
- `raw_outputs/` — raw OCR text per model per document
- `metrics/` — per-document metric JSONs
- `results/expanded_gt_metrics/` — aggregated metrics, statistical tests

## Known Limitations

1. **Consensus GT circularity**: Mistral OCR's results on financial, multi_column,
   and equations categories are inflated (CER≈0) because its output was used as GT source.
   Only its FUNSD forms results (n=5) reflect true performance.
2. **Handwritten category**: 30 docs are 32×32 DHCD character crops, not page-scale
   handwritten documents. No ground truth available for accuracy metrics.
3. **Indian languages**: Only 1 document — statistically insufficient for claims.
4. **Hardware constraints**: 16GB unified memory limits concurrent VLM execution.
   Phase 2 models (GOT-OCR, Qwen-VL, olmOCR) run sequentially.
5. **Receipt GT source**: 10 receipt ground truth files derived from Mistral OCR —
   same circularity caveat applies.
