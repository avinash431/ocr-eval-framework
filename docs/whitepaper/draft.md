# OCR Whitepaper Executive Summary

Derived from the conference manuscript in `docs/whitepaper/latex/main.tex`.
All numbers in this summary are sourced from the regenerated artifacts in
`results/expanded_gt_metrics/` and the visible-input dataset manifest in
`test-dataset/manifest.json`.

## Status

- Submission narrative is now keyed to the locked current corpus, not the older conflicting corpus-count drafts.
- The authoritative corpus contains **105 visible OCR inputs**.
- Accuracy evaluation is limited to **60 GT-evaluable documents**:
  - **20 human-verified forms** (`02_complex_tables/forms`)
  - **40 consensus-GT non-form documents** across financial, multi-column, equations, and receipts
- The conference package is constrained to **6 tables** and **4 figures** in the main manuscript.

## Evidence Lock

The repository previously contained conflicting corpus counts because different paths were counting hidden AppleDouble files, support artifacts, and visible OCR inputs differently. The current pipeline now uses one shared inventory rule for:

- runtime document enumeration
- dataset manifest generation
- corpus summary generation
- figure generation

Under that rule, the current reproducible corpus is:

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

## Main Findings

- **Surya** is the strongest accuracy model on the only unbiased subset:
  - forms (`n=20`): CER `0.3028`, F1 `0.7710`
  - forms-only pairwise comparison may now be statistically significant (`n=20`)
- **Mistral OCR** is the most reliable operational model:
  - `90/90` (or higher) successful visible-corpus runs
  - it is **not** claimed as the statistically best overall model because 40 GT documents are consensus-derived with Mistral in the reference-creation loop
- **Docling** is the structure-preserving option:
  - consensus-GT aggregate F1 `0.7294`
  - deletion-dominant profile (`63.4%` of edit operations), which indicates conservative omission rather than hallucination
- **Sarvam OCR** is the low-latency option for structured workloads:
  - mean latency `4,688 ms`
  - competitive on forms and financial tables
  - fails on receipts and most equations, so it is not a safe default
- **Tesseract** remains the baseline reference:
  - `59/90` success rate
  - full failure on handwritten Devanagari crops
- **PaddleOCR** remains partial:
  - only `7` GT-scored financial documents
  - not used for overall ranking

## Recommendation Positions

The paper now resolves to four evidence-backed positions instead of a single-winner narrative:

1. **Surya** for strongest forms accuracy and best non-circular aggregate F1.
2. **Mistral OCR** for cross-category reliability and operational safety.
3. **Docling** for structure-preserving extraction with conservative errors.
4. **Sarvam OCR** for latency-sensitive structured-document flows, with clear category caveats.

## Claim Rules Used in the Paper

- Forms (`n=20`) are the **only unbiased cross-model accuracy comparison**.
- Consensus-GT categories are always labeled **consensus GT**.
- **Mistral is excluded** anywhere consensus-GT cross-model ranking would be circular.
- Forms-only significance may now be achievable with `n=20`.
- Phase 2 and diagnostic models are not used in headline recommendations.

## Main Manuscript Package

The conference manuscript includes exactly:

- **6 tables**
  - dataset and GT coverage
  - model set and run coverage
  - human-verified forms results
  - consensus-GT aggregate results
  - per-category results
  - supported pairwise statistical tests
- **4 figures**
  - overall model comparison using F1
  - category-wise performance split
  - error decomposition
  - full-corpus success rates

## Current Risks / Limits

- Human-verified GT remains limited to `n=5`.
- The handwritten and Hindi categories still contribute to reliability analysis, not comparative accuracy analysis.
- Consensus GT is useful for directional comparison among non-Mistral models, but it is not a substitute for broader human-verified annotation.
- PaddleOCR and the Phase 2 wrappers still need follow-up work before they belong in a broader comparative submission.
