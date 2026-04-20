# OCR Model Sanity Checks — 2026-04-09

Common test input:
- `test-dataset/02_complex_tables/forms/0012199830.png`

Common command pattern:
- `source venv/bin/activate && python run_single.py --model <model> --input test-dataset/02_complex_tables/forms/0012199830.png`

## Summary table

| Model | Status | Latency | CER | WER | F1 | Notes |
|---|---|---:|---:|---:|---:|---|
| tesseract | PASS | 2896 ms | 0.5289 | 0.8072 | 0.4030 | Runs locally; baseline is noisy due to multilingual config |
| mistral_ocr | PASS | 1516 ms | 0.3347 | 0.5422 | 0.7313 | Best verified result so far on this sample |
| docling | PASS | 60519 ms | 0.4236 | 0.7831 | 0.7059 | Works, but very slow on this input; markdown-like output |
| paddleocr | PASS | 6686 ms | 0.3058 | 0.5301 | 0.7727 | Wrapper patched for installed PaddleOCR API; now strongest verified F1 on this sample |
| surya | PASS | 13437 ms | 0.3388 | 0.5542 | 0.7407 | Wrapper patched for installed Surya 0.17.x predictor API |
| sarvam_ocr | PASS | 3677 ms | 0.3430 | 0.5060 | 0.7794 | Rewritten to use Sarvam Document Intelligence SDK/job workflow |

## Raw observations

### tesseract
- Command succeeded.
- Output contained obvious script-mixing / garbling on English text.
- Current config uses `eng+hin+tel+tam+ben`, which likely hurts English-only documents.

### mistral_ocr
- Command succeeded.
- Best accuracy among verified sanity-check runs.
- Good candidate for one of the headline models in the whitepaper.

### docling
- Command succeeded.
- Preserves some document structure in markdown-like form.
- Latency is high enough that performance discussion should call this out explicitly.
- During batch evaluation, a serialization bug was discovered because `num_pages` resolved to a method object in the installed Docling version.
- Fix applied:
  - normalize `num_pages` by calling it when it is callable
  - pass `doc=result.document` into `table.export_to_dataframe(...)` to align with current Docling expectations

### paddleocr
- Command now succeeds after wrapper compatibility fix.
- Verified result:
  - latency: 6686 ms
  - CER: 0.3058
  - WER: 0.5301
  - F1: 0.7727
- Root cause:
  - the wrapper targeted an older PaddleOCR API and used incompatible init/runtime arguments (`show_log`, `use_gpu`, old `ocr(..., cls=True)` path).
- Fix applied:
  - detect supported constructor args dynamically
  - use `predict()` with `rec_texts` / `rec_scores` on newer PaddleOCR versions
  - retain fallback support for older `ocr()` output shape

### surya
- Command now succeeds after wrapper modernization.
- Verified result:
  - latency: 13437 ms
  - CER: 0.3388
  - WER: 0.5542
  - F1: 0.7407
- Root cause:
  - the wrapper targeted an older Surya API (`surya.ocr`, old model/processor import paths) that no longer exists in installed Surya 0.17.x.
- Fix applied:
  - switched to the current predictor flow using `FoundationPredictor`, `DetectionPredictor`, and `RecognitionPredictor`
  - used `TaskNames.ocr_with_boxes` with detection + recognition on the input image

### sarvam_ocr
- Command now succeeds after wrapper rewrite.
- Verified result:
  - latency: 3677 ms
  - CER: 0.3430
  - WER: 0.5060
  - F1: 0.7794
- Root cause:
  - the original wrapper assumed a nonexistent one-shot `/v1/ocr` endpoint.
  - Sarvam Vision actually uses a Document Intelligence job workflow.
- Fix applied:
  - added env overlay fix for empty YAML sections in `utils/helpers.py`
  - rewrote wrapper to use the `sarvamai` SDK:
    - create job
    - upload file
    - start job
    - wait for completion
    - download ZIP output
    - extract markdown text from archive

## Practical conclusion

As of 2026-04-09, the immediately usable sanity-checked comparison set is:
- `tesseract`
- `mistral_ocr`
- `docling`
- `paddleocr`
- `surya`
- `sarvam_ocr`

Next best expansion path:
1. Run full-dataset evaluations for the working model set first
2. Then decide whether heavier VLM models (DeepSeek, Qwen-VL, olmOCR) are worth the setup cost for Phase 1

## Recommendation for whitepaper execution

For a near-term credible draft, use:
- tesseract as baseline
- mistral_ocr as strong cloud/LLM OCR comparator
- docling as structure-aware open-source comparator

Then add:
- paddleocr after wrapper fix
- surya for multilingual coverage
