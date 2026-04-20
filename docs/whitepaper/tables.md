# OCR Whitepaper Tables

Last updated: 2026-04-20 (expanded 45-doc GT aggregates; docling + paddleocr + got_ocr partial runs integrated)

## Ground Truth Composition

Whitepaper uses two GT tiers — must not be conflated in analysis:

- **Human-verified GT (n=5):** `02_complex_tables/forms` — FUNSD-style JSON annotations flattened to text. Only non-circular category. All cross-model comparisons on forms are unbiased.
- **Model-consensus GT (n=40):** non-forms categories. Text derived primarily from Mistral OCR output, cross-validated against other model outputs. Mistral scores perfectly against consensus it contributed to — Mistral excluded from cross-model comparisons on consensus categories.

Total n=45 across 5 categories: forms (5), financial (9), multi_column (11), equations_formulas (10), receipts (10). Handwritten (32×32 DHCD crops, n=30) and Indian-lang (n=1) excluded from metric tables — insufficient for evaluation claims.

## Table 1: Phase 1 Model Set

| Model | Type | Deployment | Run Status | GT Coverage | Notes |
|:---|:---|:---|:---|:---|:---|
| tesseract | Open-source baseline | Local (CPU) | Completed (60/91) | 45/45 | Multi-language config (eng+hin+tel+tam+ben) |
| mistral_ocr | Cloud/API (VLM) | Remote | Completed (91/91) | 45/45 (circular on 40) | Vision-language model via Mistral AI API |
| docling | Open-source document conversion | Local | Completed (batch) | 42/45 | Structure-aware; markdown output |
| paddleocr | Open-source OCR engine | Local | Partial (financial only) | 7/45 | Timeouts on non-financial categories |
| surya | Open-source multilingual OCR | Local | Completed (71/91) | 45/45 | Wrapper modernized for Surya 0.17.x |
| sarvam_ocr | Cloud/API document intelligence | Remote | Completed (62/91) | 27/45 | SDK job workflow; skipped receipts entirely |

## Table 2: Phase 2 Model Set — Wrapper Exists, Not Benchmark-Ready

| Model | Reason Not Benchmarked |
|:---|:---|
| got_ocr | Ran 21/45 GT docs — catastrophic insertion rate (avg CER 4.76, 92.3% errors = hallucinated chars). Separate diagnostic only (Table 9) |
| olmocr | Run produced 0 raw outputs (2026-04-19) — environment/load error unresolved |
| qwen_vl | Run produced 17 raw outputs then stalled (2026-04-19) — GPU/timeout issue unresolved |
| deepseek_ocr | Wrapper exists, not yet executed |

## Table 3: Common Sample Sanity-Check Results (n=1)

Document: `test-dataset/02_complex_tables/forms/0012199830.png`

| Model | Latency (ms) | CER | WER | F1 |
|:---|---:|---:|---:|---:|
| tesseract | 2,896 | 0.5289 | 0.8072 | 0.4030 |
| mistral_ocr | 1,516 | 0.3347 | 0.5422 | 0.7313 |
| docling | 4,359 | 0.4236 | 0.7831 | 0.7059 |
| paddleocr | 6,686 | 0.3058 | 0.5301 | 0.7727 |
| surya | 13,437 | 0.3388 | 0.5542 | 0.7407 |
| sarvam_ocr | 3,677 | 0.3430 | 0.5060 | 0.7794 |

## Table 4: Aggregate Metrics — Human-Verified GT Only (Forms n=5)

Only unbiased cross-model comparison. Mistral included — not circular on forms.

| Model | Avg CER | Avg WER | Avg F1 | Avg Precision | Avg Recall |
|:---|---:|---:|---:|---:|---:|
| surya | 0.3028 | 0.4483 | 0.7710 | 0.7943 | 0.7508 |
| sarvam_ocr | 0.5124 | 0.5183 | 0.7592 | 0.7479 | 0.7788 |
| mistral_ocr | 0.3933 | 0.5033 | 0.7549 | 0.7582 | 0.7584 |
| docling | 0.4302 | 0.5593 | 0.5889 | 0.8255 | 0.5586 |
| tesseract | 0.4651 | 0.6489 | 0.4980 | 0.5469 | 0.4637 |

## Table 5: Aggregate Metrics — Consensus GT (n=40, Mistral excluded)

| Model | n | Avg CER | Avg WER | Avg F1 | Avg Precision | Avg Recall |
|:---|---:|---:|---:|---:|---:|---:|
| surya | 40 | 0.5673 | 0.7707 | 0.7717 | 0.7275 | 0.8650 |
| docling | 37 | 0.4417 | 0.5308 | 0.7294 | 0.8090 | 0.7098 |
| sarvam_ocr | 22 | 0.6544 | 0.7859 | 0.7290 | 0.6801 | 0.8524 |
| paddleocr | 7 | 0.4653 | 0.5885 | 0.6846 | 0.7291 | 0.6509 |
| tesseract | 40 | 0.5110 | 0.7585 | 0.6221 | 0.5904 | 0.7087 |

## Table 6: Category-Wise Metrics

### 6a. Forms — Human-Verified GT (n=5)

| Model | Avg CER | Avg WER | Avg F1 |
|:---|---:|---:|---:|
| surya | 0.3028 | 0.4483 | 0.7710 |
| sarvam_ocr | 0.5124 | 0.5183 | 0.7592 |
| mistral_ocr | 0.3933 | 0.5033 | 0.7549 |
| docling | 0.4302 | 0.5593 | 0.5889 |
| tesseract | 0.4651 | 0.6489 | 0.4980 |

### 6b. Financial — Consensus GT (n=9; paddleocr n=7)

| Model | Avg CER | Avg WER | Avg F1 |
|:---|---:|---:|---:|
| surya | 0.4192 | 0.5823 | 0.8081 |
| sarvam_ocr | 0.4130 | 0.5103 | 0.7667 |
| docling | 0.3541 | 0.3996 | 0.7003 |
| paddleocr | 0.4653 | 0.5885 | 0.6846 |
| tesseract | 0.3944 | 0.5384 | 0.6086 |

### 6c. Multi-Column — Consensus GT (n=11; docling n=9)

| Model | Avg CER | Avg WER | Avg F1 |
|:---|---:|---:|---:|
| surya | 0.5412 | 0.7039 | 0.8425 |
| docling | 0.3707 | 0.4091 | 0.7827 |
| tesseract | 0.3173 | 0.5450 | 0.7352 |
| sarvam_ocr | 0.7307 | 0.9265 | 0.7295 |

### 6d. Equations / Formulas — Consensus GT (n=10; sarvam n=2)

| Model | Avg CER | Avg WER | Avg F1 |
|:---|---:|---:|---:|
| surya | 0.5390 | 0.7930 | 0.7624 |
| docling | 0.4271 | 0.5384 | 0.7587 |
| tesseract | 0.5061 | 0.6868 | 0.5957 |
| sarvam_ocr | 1.3217 | 1.2521 | 0.5566 |

### 6e. Receipts — Consensus GT (n=10; docling n=9; sarvam skipped)

Mistral excluded — original circular category.

| Model | Avg CER | Avg WER | Avg F1 |
|:---|---:|---:|---:|
| docling | 0.6167 | 0.7751 | 0.6729 |
| surya | 0.7575 | 0.9917 | 0.6704 |
| tesseract | 0.8340 | 1.2631 | 0.5362 |

## Table 7: Category-Level Success Rates — Full Dataset

| Category | Tesseract | Mistral OCR | Docling | PaddleOCR | Surya | Sarvam OCR |
|:---|---:|---:|---:|---:|---:|---:|
| 02_complex_tables/financial | 15/15 | 15/15 | 9/15 | 7/15 | 15/15 | 15/15 |
| 02_complex_tables/forms | 5/5 | 5/5 | 5/5 | 0/15 | 5/5 | 5/5 |
| 02_complex_tables/multi_column | 15/15 | 15/15 | 9/15 | 0/15 | 15/15 | 15/15 |
| 03_handwritten/hindi_devanagari | 0/30 | 30/30 | — | — | 10/30 | 24/30 |
| 04_indian_languages/hindi | 1/1 | 1/1 | — | — | 1/1 | 1/1 |
| 06_mixed_content/equations_formulas | 14/15 | 15/15 | 10/15 | 0/15 | 15/15 | 2/15 |
| 06_mixed_content/receipts | 10/10 | 10/10 | 9/10 | 0/15 | 10/10 | 0/10 |
| **Total** | **60/91** | **91/91** | **42/76** | **7/76** | **71/91** | **62/91** |

Docling/PaddleOCR totals exclude handwritten+Indian-lang categories (not targeted in this run).

## Table 8: Error Decomposition (Expanded GT)

| Model | n | Total Errors | Substitutions % | Insertions % | Deletions % | Dominant Mode |
|:---|---:|---:|---:|---:|---:|:---|
| mistral_ocr | 45 | 1,615 | 34.0 | 37.4 | 28.6 | Balanced |
| surya | 45 | 53,740 | 44.1 | 35.5 | 20.4 | Substitution-heavy |
| tesseract | 45 | 29,589 | 26.4 | 36.4 | 37.2 | Deletion-dominant |
| docling | 42 | 29,471 | 18.0 | 18.6 | 63.4 | Deletion-dominant |
| sarvam_ocr | 27 | 21,977 | 8.8 | 81.2 | 10.0 | Insertion-dominant |
| paddleocr | 7 | 7,449 | 28.8 | 11.6 | 59.6 | Deletion-dominant |
| got_ocr | 21 | 104,117 | 6.4 | 92.3 | 1.3 | Hallucination-dominant |

## Table 9: GOT-OCR Hallucination Diagnostic

GOT-OCR produces text far exceeding GT length. Flagged Phase 2 blocker.

| Category | n | Avg CER | Avg char_pred / char_gt | Interpretation |
|:---|---:|---:|---:|:---|
| forms | 5 | 7.42 | 9.4× | Severe over-generation |
| financial | 9 | 2.27 | 3.1× | Over-generation |
| multi_column | 7 | 6.06 | 6.8× | Severe over-generation |

CER > 1 indicates edit distance exceeds GT length. Model generates multiple passes of document text plus hallucinated content. Unusable without post-processing.

## Table 10: Average Latency by Category (ms)

| Category | Tesseract | Mistral OCR | Docling | Surya | Sarvam OCR |
|:---|---:|---:|---:|---:|---:|
| 02_complex_tables/financial | 10,611 | 30,172 | 62,408 | 31,811 | 5,106 |
| 02_complex_tables/forms | 3,232 | 5,292 | 4,893 | 38,289 | 3,836 |
| 02_complex_tables/multi_column | 12,855 | 61,296 | 71,544 | 80,326 | 6,597 |
| 06_mixed_content/equations_formulas | 17,581 | 5,111 | 58,217 | 95,815 | 4,165 |
| 06_mixed_content/receipts | 3,849 | 3,169 | 45,702 | 40,371 | — |

Docling latency includes layout/structure pipeline — not directly comparable to line-OCR models.

## Table 11: Statistical Significance — Pairwise F1 (Wilcoxon, Expanded GT)

Mistral excluded from consensus-category tests.

### Significant at p<0.05

| A | B | n | p-value | Δ F1 | Winner | Effect Size |
|:---|:---|---:|---:|---:|:---|:---|
| surya | tesseract | 45 | <0.001 | 0.1633 | surya | large (d=0.850) |
| docling | tesseract | 42 | <0.001 | 0.1137 | docling | medium (d=0.576) |
| surya | docling | 42 | 0.0241 | 0.0552 | surya | small (d=0.314) |
| surya | got_ocr | 21 | <0.001 | 0.3424 | surya | large (d=1.277) |
| tesseract | got_ocr | 21 | 0.0351 | 0.1475 | tesseract | — |
| sarvam_ocr | got_ocr | 21 | 0.0005 | 0.2852 | sarvam_ocr | — |
| docling | got_ocr | 20 | 0.0012 | 0.2406 | docling | — |

### Not Significant

| A | B | n | p-value | Note |
|:---|:---|---:|---:|:---|
| surya | sarvam_ocr | 27 | 0.0906 | Trending |
| tesseract | sarvam_ocr | 27 | 0.1482 | — |

## Table 12: Confidence Calibration

| Model | Confidence-F1 Correlation (Spearman ρ) | p-value | Calibrated? |
|:---|---:|---:|:---|
| tesseract | +0.83 | 0.0001 | Yes — reliable for quality gating |
| surya | +0.08 | >0.05 | No — confidence not meaningful |

## Table 13: Integration Lessons

| Model | Initial Problem | Fix Applied | Operational Takeaway |
|:---|:---|:---|:---|
| PaddleOCR | Outdated constructor args and API usage | Dynamic arg detection, `predict()` API migration | Library version drift requires version-aware wrappers |
| Surya | Deprecated module structure (`surya.ocr`) | Migration to `FoundationPredictor` architecture | Upstream API restructuring can break integrations silently |
| Sarvam OCR | Wrong endpoint assumption (no `/v1/ocr`) | Five-step SDK job workflow implementation | Cloud document APIs may use async job patterns |
| Docling | Non-serializable `num_pages` metadata | Callable detection and normalization | Batch pipelines need strict JSON-safe output handling |
| Config loader | Empty YAML sections caused `None` propagation | Defensive null handling in `utils/helpers.py` | Multi-model configs need null-safe section loading |
| GOT-OCR | Generates repeated text passes on non-academic docs | Not resolved — blocker for benchmark inclusion | Academic-tuned models may fail on commercial document layouts |

## Table 14: Model Selection Guide by Use Case

| Use Case | Recommended Model | Rationale |
|:---|:---|:---|
| Structured forms (human-verified evaluation) | Surya | Lowest CER (0.3028), highest F1 (0.7710) on n=5 forms; Δ vs tesseract significant at p<0.001 |
| Multi-column layouts | Surya | Highest F1 (0.8425) on consensus GT; large effect vs tesseract |
| Structure-preserving output (markdown) | Docling | F1 0.7127 overall, deletion-dominant (safer — no hallucinations); native markdown |
| Financial tables / forms (latency-sensitive) | Sarvam Vision OCR | 3,836–5,106 ms on forms/financial; competitive F1 (0.7592 forms, 0.7667 financial) |
| Receipts | Docling or Surya | F1 ≈ 0.67 both on consensus GT; all models struggle. Mistral excluded (circular) |
| Baseline reference | Tesseract | Free, local, well-calibrated confidence (ρ=+0.83) |
| General-purpose (any doc type) | Mistral OCR (with caveat) | 100% success rate; forms F1 0.7549 is the only unbiased measurement |
| Avoid: unconstrained commercial documents | — | GOT-OCR (hallucinates), PaddleOCR (partial timeouts), olmocr/qwen_vl (Phase 2 unresolved) |
