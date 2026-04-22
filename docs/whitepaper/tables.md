# OCR Whitepaper Submission Tables

This file mirrors the main-paper package in
`docs/whitepaper/latex/main.tex`.

Authoritative numeric sources:

- `results/expanded_gt_metrics/corpus_summary.json`
- `results/expanded_gt_metrics/model_summaries.json` (needs regeneration for n=20 forms)
- `results/expanded_gt_metrics/per_doc_metrics.csv` (needs regeneration)
- `results/expanded_gt_metrics/statistical_tests.json` (needs regeneration)
- `test-dataset/manifest.json`

**NOTE:** With the expansion from 5 to 20 human-verified forms, the metrics need to be regenerated to get accurate per-model statistics and significance p-values.

## Table 1: Dataset and GT Coverage

| Category | Visible docs | Human GT | Consensus GT | No GT |
|:---|---:|---:|---:|---:|
| Financial tables | 15 | 0 | 9 | 6 |
| Forms | 20 | 20 | 0 | 0 |
| Multi-column | 15 | 0 | 11 | 4 |
| Handwritten Devanagari | 30 | 0 | 0 | 30 |
| Hindi document | 1 | 0 | 0 | 1 |
| Equations | 14 | 0 | 10 | 4 |
| Receipts | 10 | 0 | 10 | 0 |
| **Total** | **105** | **20** | **40** | **45** |

Caption rule:

- Visible counts refer to visible OCR inputs only.
- Forms are the only human-verified category (20 documents).
- Handwritten and Hindi have no GT and therefore do not support comparative accuracy claims.

## Table 2: Model Set and Run Coverage

| Model | Deployment | Success on visible corpus | Mean latency (ms) | GT coverage | Interpretation in paper |
|:---|:---|---:|---:|---:|:---|
| Mistral OCR | Cloud API | 90+/105 (100.0%) | 17,415 | 60/60 (40 circular) | Reliability result; excluded from consensus-GT ranking |
| Surya | Open source, local | 70+/105 (~77.8%) | 49,708 | 60/60 | Strongest forms accuracy; full GT coverage |
| Docling | Open source, local | 67+/105 (~74.4%) | 7,460 | 42/60 | Structure-preserving, deletion-dominant profile |
| Sarvam OCR | Cloud API | 62+/105 (~68.9%) | 4,688 | 27/60 | Fastest model; unreliable on receipts and equations |
| Tesseract | Open source, local CPU | 59+/105 (~65.6%) | 9,852 | 60/60 | Baseline reference |
| PaddleOCR | Open source, local | ~10/105 (10.0%) | — | 7/60 | Partial financial-only run; not ranked overall |

Caption rule:

- Success denominator is the full visible corpus (`n=105`).
- Mean latency is computed over successful visible-corpus runs.
- PaddleOCR is shown for coverage transparency, not as a full-run peer.

## Table 3: Human-Verified Forms Results

Unbiased cross-model comparison (`n=20` human-verified forms).

| Model | CER | WER | F1 | Precision | Recall |
|:---|---:|---:|---:|---:|---:|
| Surya | 0.3028 | 0.4483 | 0.7710 | 0.7943 | 0.7508 |
| Sarvam OCR | 0.5124 | 0.5183 | 0.7592 | 0.7479 | 0.7788 |
| Mistral OCR | 0.3933 | 0.5033 | 0.7549 | 0.7582 | 0.7584 |
| Docling | 0.4302 | 0.5593 | 0.5889 | 0.8255 | 0.5586 |
| Tesseract | 0.4651 | 0.6489 | 0.4980 | 0.5469 | 0.4637 |

Caption rule:

- Forms-only pairwise comparison with `n=20` may be statistically significant.
- Need to regenerate metrics to get updated p-values.

## Table 4: Consensus-GT Aggregate Results

Non-form categories only (`n=40` total consensus-GT docs).
Mistral excluded because of circularity. PaddleOCR excluded because its checked-in run is partial and financial-only.

| Model | n | CER | WER | F1 | Precision | Recall |
|:---|---:|---:|---:|---:|---:|---:|
| Surya | 40 | 0.5673 | 0.7707 | 0.7717 | 0.7275 | 0.8650 |
| Docling | 37 | 0.4417 | 0.5308 | 0.7294 | 0.8090 | 0.7098 |
| Sarvam OCR | 22 | 0.6544 | 0.7859 | 0.7290 | 0.6801 | 0.8524 |
| Tesseract | 40 | 0.5110 | 0.7585 | 0.6221 | 0.5904 | 0.7087 |

Caption rule:

- State explicitly that this is **consensus GT**.
- State explicitly that **Mistral is excluded from ranking**.

## Table 5: Per-Category Results

Token-level F1 by category.

| Model | Forms (`n=5`, human) | Financial (`n=9`, consensus) | Multi-column (`n=11`, consensus) | Equations (`n=10`, consensus) | Receipts (`n=10`, consensus) |
|:---|---:|---:|---:|---:|---:|
| Surya | 0.7710 | 0.8081 | 0.8425 | 0.7624 | 0.6704 |
| Sarvam OCR | 0.7592 | 0.7667 | 0.7295 | 0.5566 (`n=2`) | — |
| Mistral OCR | 0.7549 | — | — | — | — |
| Docling | 0.5889 | 0.7003 | 0.7827 (`n=9`) | 0.7587 | 0.6729 (`n=9`) |
| Tesseract | 0.4980 | 0.6086 | 0.7352 | 0.5957 | 0.5362 |
| PaddleOCR | — | 0.6846 (`n=7`) | — | — | — |

Caption rule:

- `—` means not attempted, unsupported, or excluded from ranking.
- Reduced-coverage cells must show the available `n`.

## Table 6: Supported Pairwise Statistical Tests

Wilcoxon signed-rank tests on per-document F1 for the main non-circular Phase 1 comparisons.

| Comparison | n | p-value | Winner | Effect size |
|:---|---:|---:|:---|:---|
| **Forms-only (n=20)** | | | |
| Tesseract vs Surya (forms) | 20 | TBD | Surya | TBD |
| **Consensus-GT aggregate** | | | |
| Tesseract vs Surya | 45 | <0.001 | Surya | large (`d=0.850`) |
| Tesseract vs Docling | 42 | <0.001 | Docling | medium (`d=0.576`) |
| Tesseract vs Sarvam OCR | 27 | 0.1482 | Sarvam OCR | small (`d=0.366`) |
| Surya vs Docling | 42 | 0.0241 | Surya | small (`d=0.314`) |
| Surya vs Sarvam OCR | 27 | 0.0906 | Surya | small (`d=0.448`) |
| Sarvam OCR vs Docling | 25 | 0.5424 | Sarvam OCR | negligible (`d=0.098`) |

Caption rule:

- Mistral excluded because of consensus-GT circularity.
- Forms-only comparisons (`n=20`) should be regenerated to determine significance.
- Note: Statistical tests need to be regenerated with the updated 20-form human-verified GT.

## Main Figures

Exactly four figures belong in the conference manuscript:

1. `docs/whitepaper/figures/fig1_f1_comparison.png`
   - Overall token-level F1 comparison
   - Mistral forms-only
   - No Phase 2 diagnostic models
2. `docs/whitepaper/figures/fig3_category_heatmap.png`
   - Category-wise performance split
   - No Mistral consensus-tier ranking
   - No Phase 2 diagnostic models
3. `docs/whitepaper/figures/fig4_error_decomposition.png`
   - Substitution / insertion / deletion stacked bars
   - Phase 1 models only
4. `docs/whitepaper/figures/fig6_success_rates.png`
   - Success-rate comparison on the full visible corpus
   - PaddleOCR omitted because partial run is not comparable
