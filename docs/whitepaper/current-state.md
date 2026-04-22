# OCR Whitepaper Current State

Chosen working scope recommendation: Phase 1 internal whitepaper with early benchmark findings, methodology, framework design, and a smaller evidence-backed model comparison.

Date assessed: 2026-04-09
Repo: `ocr-eval-framework`

## Framework status

The framework is real and runnable.

Verified facts:
- `python cli/run_batch.py --list` successfully loads 10 registered OCR model wrappers:
  - deepseek_ocr
  - docling
  - got_ocr
  - mistral_ocr
  - olmocr
  - paddleocr
  - qwen_vl
  - sarvam_ocr
  - surya
  - tesseract
- Local baseline command works:
  - `source venv/bin/activate && python cli/run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png`
- Verified output from that baseline run:
  - ~495 extracted characters
  - latency ~2918 ms in the latest run
  - CER 0.5289
  - WER 0.8072
  - F1 0.4030

Interpretation:
- The evaluation framework is mature enough to support a paper.
- The whitepaper is currently blocked more by missing experiment coverage and drafting than by missing framework code.

## Dataset coverage

Current committed dataset inventory: 101 documents.

Observed category distribution:
- `02_complex_tables`: 45 docs
  - financial: 15
  - forms: 15
  - multi_column: 15
- `03_handwritten`: 30 docs
  - hindi_devanagari: 30
- `04_indian_languages`: 1 doc
  - hindi: 1
- `06_mixed_content`: 25 docs
  - equations_formulas: 15
  - receipts: 10

Important limitation:
- The committed dataset is narrower than the original whitepaper plan suggests.
- It does not currently support strong claims about broad multilingual coverage, broad European-language coverage, or substantial internal-document coverage unless those assets exist elsewhere and are added/evaluated.

## Existing results

The tracker workbook currently shows only two logged model runs:

1. Tesseract
- Document: `0012199830.png`
- Category: Complex Tables/Forms
- CER: 0.5289
- WER: 0.8072
- F1: 0.4030
- Latency: 4111 ms
- Note: garbled Indic characters due to multi-language config

2. Mistral OCR
- Document: `0012199830.png`
- Category: Complex Tables/Forms
- CER: 0.3347
- WER: 0.5422
- F1: 0.7313
- Latency: 1758 ms
- Note: much better accuracy on chemical names and company name

Important limitation:
- There is no existing batch results directory from a prior full comparison run.
- The paper currently does not have enough evidence for strong cross-model conclusions.

## Risks to paper credibility

1. Scope-evidence mismatch
- The outline describes a broad 14-model enterprise benchmark across many document types and languages.
- The current tracked evidence is only a tiny subset of that ambition.

2. Dataset mismatch
- The planned categories in the outline/tracker are broader than the actual committed dataset.

3. Measurement mismatch
- The code automates CER, WER, BLEU, edit distance, F1, and latency.
- It does not yet appear to automatically measure all claimed dimensions like privacy, deployment maturity, integration quality, or full structured extraction quality.
- Those dimensions need either manual scoring criteria or narrower claims.

4. Recommendation risk
- A strong “best overall OCR model” recommendation would currently be under-supported.
- Safer recommendation format: best among tested subset, plus explicit next-phase evaluation roadmap.

## Recommended next action

Recommended path: finish this as a Phase 1 internal whitepaper.

That means:
- emphasize methodology and framework quality
- run 3-5 models well rather than promising all 14 immediately
- make evidence-backed recommendations only for the tested subset
- include a clearly labeled roadmap for remaining models/categories

## Suggested minimum viable model set

Priority run order:
1. tesseract
2. mistral_ocr
3. paddleocr
4. docling
5. surya

Optional expansions after that:
- sarvam_ocr
- qwen_vl
- olmocr

## Working thesis for the paper

Instead of claiming a finished universal benchmark, position the paper as:

“A practical internal OCR evaluation framework and Phase 1 benchmark, comparing a representative subset of OCR systems across complex forms, handwritten Hindi, mixed-content documents, and selected enterprise use cases.”

That framing is truthful, useful, and finishable.
