# OCR Evaluation Framework — Project Memory

## Project Identity
Internal benchmark whitepaper evaluating open-source OCR systems.
Phase 1 covers 6 working models on 101 documents across 4 document categories.
Framework scope: open-source and API-accessible OCR models only (no AWS, Azure, or GCP).

**Non-negotiable principle**: All whitepaper claims must align with actual
benchmark evidence from `results/`. Never over-claim. Never present Phase 2
models as part of current findings.

## Architecture
- `models/`        — Self-registering OCR wrappers via `@register_model`. All inherit `BaseOCRModel` and implement `_ocr_impl()`
- `utils/`         — Runner, metrics (CER/WER/F1/edit_dist/precision/recall/word_accuracy/error_decomposition), HTML report generation, helpers
- `configs/`       — YAML for non-secret settings only. All API keys go in `.env`, never in YAML or code
- `docs/whitepaper/` — draft.md, tables.md, sanity-checks, current-state (writing artifacts)
- `docs/plans/`    — Revival plan and milestone tracking
- `results/`       — Timestamped batch dirs; source of truth for all benchmark claims
- `test-dataset/`  — 101 documents across 02_complex_tables, 03_handwritten, 04_indian_languages, 06_mixed_content

## Model Registry (10 models — open-source only)

### Phase 1 — Working and sanity-checked
| Model         | Type            | Notes                                      |
|---------------|-----------------|---------------------------------------------|
| tesseract      | Local baseline  | Weakest accuracy; well-calibrated confidence (rho=+0.83) |
| mistral_ocr    | API             | 100% success rate; strongest generalization |
| docling        | Open-source     | Structure-aware; slower runtime (>60s/doc)  |
| paddleocr      | Open-source     | Strong sanity-check; timeout on batch       |
| surya          | Open-source     | Lowest forms CER (0.3028); highest F1 (0.7710) |
| sarvam_ocr     | API             | Lowest latency (4,688ms); competitive F1    |

### Phase 2 — Wrappers exist, not yet benchmarked
deepseek_ocr, qwen_vl, olmocr, got_ocr

## Key Commands
```bash
source venv/bin/activate
python cli/run_single.py --model X --input Y     # single model, single doc
python cli/run_model.py --model X                # single model, full dataset
python cli/run_batch.py                          # all working models, full dataset
python cli/evaluate.py --results-dir Z           # generate HTML report + summary table
python cli/evaluate.py --results-dir Z --export-csv  # also export metrics CSV
python tools/download_dataset.py                   # download test documents
```

## Metrics Supported
CER (lower=better), WER (lower=better), word accuracy (higher=better),
normalized edit distance (lower=better), token F1 with precision/recall (higher=better),
error decomposition (substitution/insertion/deletion rates), latency ms.

Text normalization applied before all metrics: Unicode NFKC, markdown/HTML stripping,
whitespace collapse. F1 uses Counter-based bag-of-words (not set-based).

## Known Data Caveats
- Receipt ground truth derived from Mistral OCR output — creates circularity (Mistral gets CER=0, F1=1.0 on receipts)
- Handwritten category is 32x32 DHCD character crops, not page-scale handwritten documents
- Indian languages category has only 1 document — statistically insufficient
- Ground truth covers only 15/91 docs (5 forms + 10 receipts)
- No pairwise F1 differences are statistically significant at n=5 (Wilcoxon p>0.05)

## Active Batch Run
`results/20260411_030236_batch/` — tesseract + mistral_ocr full results with corrected metrics

## Conventions
- API keys in `.env` only — never in YAML or code
- Ground truth files use `<stem>_gt.txt` naming convention
- New models: create file in `models/` — self-registration, no manual wiring needed
- Results saved to timestamped dirs under `results/`
- Whitepaper writing artifacts live in `docs/whitepaper/`

## Adding a New Model
1. Create `models/<model_name>.py`
2. Inherit `BaseOCRModel`, implement `_ocr_impl(image_path) -> str`
3. Decorate class with `@register_model`
4. Add non-secret config to `configs/config.yaml`
5. Add API key reference to `.env.example`

## Import Shortcuts
See @docs/whitepaper/current-state.md for verified dataset and model status
See @docs/plans/2026-04-09-ocr-whitepaper-revival-plan.md for phase milestones
