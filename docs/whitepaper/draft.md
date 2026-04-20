# A Practical OCR Evaluation Framework and Phase 1 Benchmark: Comparing Six OCR Systems Across Complex Enterprise Documents

**Draft status:** Working draft — updated 2026-04-20. Full-dataset results included for Tesseract, Mistral OCR, Surya, Sarvam Vision OCR, and Docling. PaddleOCR partial (financial category only). Ground-truth corpus expanded from 15 to 45 documents: 5 human-verified forms and 40 model-consensus annotations. GOT-OCR ran but produced unusable hallucinated output — reported as Phase 2 diagnostic only. Metrics use corrected pipeline (Counter-based F1, Unicode NFKC normalization).

---

## 1. Executive Summary

This whitepaper presents a practical, reproducible evaluation framework for comparing optical character recognition (OCR) systems across a curated corpus of 101 enterprise-relevant documents. The evaluation covers four document categories — complex tables and forms, handwritten Devanagari character samples, Indian-language material, and mixed-content documents including receipts and mathematical formulas — and measures seven automated metrics: character error rate (CER), word error rate (WER), word accuracy, normalized edit distance, token-level F1 with precision and recall, and per-document latency. Error decomposition into substitution, insertion, and deletion rates provides diagnostic insight into failure modes.

Phase 1 of this benchmark integrates and evaluates six OCR systems: Tesseract as a local CPU baseline, Mistral OCR as a cloud-hosted vision-language model, Docling as a structure-aware document converter, PaddleOCR as a high-performance open-source engine, Surya as a multilingual OCR toolkit, and Sarvam Vision OCR as an India-focused document intelligence service. Each system was integrated into the framework through a unified wrapper interface that required resolving non-trivial API compatibility and integration challenges, a process that itself yields valuable operational insights for enterprise adoption decisions.

Full-dataset evaluation of five models on an expanded 45-document ground-truth subset reveals meaningful performance differences. The ground-truth corpus comprises 5 human-verified forms and 40 model-consensus annotations across financial tables, multi-column layouts, equations/formulas, and receipts. Mistral OCR achieved the highest reliability (100% success rate across all document categories); its overall F1 of 0.9728 is inflated by consensus-GT circularity on 40 of 45 documents and is not used for cross-model ranking. On the unbiased human-verified forms subset, Surya delivered the lowest CER (0.3028) with the strongest F1 (0.7710); Mistral, Sarvam, and Surya cluster within 0.02 F1 of each other at this sample size. Pairwise Wilcoxon tests on the expanded consensus subset confirm statistically significant F1 advantages for Surya over Tesseract (p<0.001, large effect) and Docling over Tesseract (p<0.001, medium effect). Surya over Docling is significant at p=0.024 (small effect). Docling completed its first full batch run, scoring F1 0.7294 on consensus GT with a deletion-dominant error profile (63.4%) indicating conservative output — no hallucination. Sarvam Vision OCR delivered the lowest latency (4,688 ms) but failed on receipts entirely. Tesseract established the baseline with 65.9% success rate and weakest accuracy. GOT-OCR produced catastrophic hallucination (CER 4.76, 92.3% insertion rate) and is excluded from findings, flagged as a Phase 2 blocker. OlmOCR and Qwen-VL runs failed in the current environment and remain unresolved. Error decomposition reveals four distinct failure modes across Phase 1 models: balanced (Mistral), substitution-heavy (Surya), deletion-dominant (Docling, PaddleOCR, Tesseract), and insertion-dominant (Sarvam). The framework is positioned as a reusable benchmarking asset, with a Phase 2 roadmap covering human-verified GT expansion and unresolved VLM integrations.

## 2. Business Context and Motivation

Enterprise workflows that depend on extracting text and structure from scanned documents, forms, reports, receipts, and multilingual material require OCR systems that deliver reliable accuracy under diverse conditions. Typical use cases include form digitization for regulatory compliance, invoice and receipt extraction for accounts payable automation, conversion of scanned archives into searchable repositories, multilingual document processing for global operations, and structured extraction from tables and semi-structured layouts for data pipelines.

The practical challenge confronting enterprise architects is that OCR systems differ significantly across multiple dimensions that are difficult to evaluate from vendor documentation alone. Text accuracy varies substantially depending on document type, script, and image quality. Latency and throughput profiles differ between local and cloud-hosted systems in ways that affect pipeline design. Table and layout preservation capabilities range from absent to sophisticated. Multilingual support — particularly for Indian languages such as Hindi, Tamil, Telugu, and Bengali — remains unevenly developed across the OCR ecosystem. Operational complexity, including API stability, dependency management, and deployment mode, adds further decision weight that pure accuracy benchmarks do not capture.

This evaluation was motivated by the observation that no single publicly available benchmark adequately addresses the intersection of these concerns for an enterprise GenAI practice team evaluating OCR systems for production adoption. The framework described here is designed to produce evidence-backed recommendations grounded in reproducible measurements rather than vendor claims.

## 3. Scope of This Evaluation

This whitepaper is intentionally framed as a Phase 1 internal benchmark. The original project plan envisioned a broader 14-model comparison across a wider multilingual dataset; the currently verified evidence base is narrower and more defensible. This paper prioritizes accuracy of claims over breadth of ambition, and all quantitative findings should be read in light of the dataset and model coverage described below.

The evaluation corpus consists of 101 committed test documents distributed across four categories. The complex tables category contains 45 documents spanning financial tables, structured forms, and multi-column layouts. The handwritten category contains 30 images of isolated Devanagari characters sourced from the DHCD (Devanagari Handwritten Character Dataset), each a 32x32 pixel grayscale image depicting a single character class. These samples test character-level recognition capability but do not represent document-scale handwriting recognition; they are included to stress-test model robustness on minimal-context inputs rather than to evaluate full handwritten document OCR. The Indian languages category contains a single Hindi document, which is acknowledged as statistically insufficient for drawing language-specific conclusions. The mixed-content category contains 25 documents including 15 with mathematical equations and formulas and 10 receipts.

Ground truth annotations are available for 45 documents across five subcategories: 5 forms (human-verified FUNSD-derived JSON flattened to text), 9 financial tables, 11 multi-column layouts, 10 equations/formulas, and 10 receipts. The 40 non-forms annotations were generated by a model-consensus procedure: primary candidate text from the highest-accuracy sanity-check model (Mistral OCR) cross-validated against other model outputs (see `generate_ground_truth.py`). This distinction matters for interpretation — cross-model comparisons against consensus GT are biased in favor of Mistral because its own output contributed to the reference. Accordingly, this paper reports two evidence tiers: unbiased human-verified GT (forms only, n=5) and consensus GT (non-forms, n=40) with Mistral excluded from cross-model ranking on the latter. Metrics are computed only where ground truth exists, and every cross-model comparison in this paper declares the GT tier it uses.

It is important to note that the committed dataset does not currently support strong claims about broad multilingual coverage or comprehensive European-language evaluation. The Indian languages category is critically under-represented with a single document, and any multilingual observations should be treated as preliminary rather than statistically validated.

## 4. Evaluation Framework Architecture

The benchmark is implemented in the `ocr-eval-framework` repository as a modular Python application designed for extensibility and reproducibility. The architecture separates concerns into four layers: model wrappers, execution orchestration, metrics computation, and reporting.

### 4.1 Model Integration Layer

All OCR model wrappers inherit from a common `BaseOCRModel` abstract class and implement a single `_ocr_impl(image_path) -> str` method. Registration is automatic via a `@register_model` decorator — adding a new model requires only creating a Python file in the `models/` directory with no manual wiring to the framework. At the time of writing, the registry contains 10 open-source and API-accessible model wrappers, of which six have been fully integrated and sanity-checked for Phase 1.

Each wrapper returns a standardized `OCRResult` object containing the extracted text, structured data where available, confidence scores, latency in milliseconds, estimated cost, error information, and success status. This normalization enables fair cross-model comparison regardless of the underlying API shape.

### 4.2 Execution Engine

The `EvalRunner` class provides three execution modes. Single-document mode (`run_single.py`) processes one document through one model for rapid testing and debugging. Model-level mode (`run_model.py`) runs one model across the entire dataset. Batch mode (`run_batch.py`) runs multiple models sequentially across the full dataset, producing a unified results directory with per-model output files, per-model metrics files, and a batch summary JSON.

All results are saved to timestamped directories under `results/`, ensuring that successive evaluation runs do not overwrite prior artifacts. Raw OCR outputs are preserved alongside computed metrics for post-hoc failure analysis.

### 4.3 Metrics Pipeline

The metrics module (`utils/metrics.py`) applies a normalization pipeline before computing metrics, ensuring fair comparison across models that produce different output formats. Normalization consists of Unicode NFKC normalization (critical for Devanagari and accented characters), stripping of markdown and HTML formatting artifacts (headings, bold/italic markers, table syntax, links, images), and whitespace collapse to single spaces. This normalization addresses the systematic bias that would otherwise inflate error rates for models like Mistral OCR and Docling that produce markdown-formatted output.

Seven automated measures are computed for each document where ground truth is available. Character Error Rate (CER) and Word Error Rate (WER) are computed using Levenshtein distance via the `rapidfuzz` library, normalized by ground truth length. Word accuracy is defined as max(1 - WER, 0). Normalized edit distance provides a 0-to-1 scale where 0 indicates identical strings, normalized by the maximum of prediction and ground truth lengths. Token-level F1 uses Counter-based bag-of-words matching with multiplicity — unlike set-based F1, this correctly penalizes missing or duplicated tokens and reflects the actual word frequency distribution. Precision and recall are reported separately alongside F1 to distinguish models that over-generate text (low precision, high recall) from those that under-extract (high precision, low recall). Latency is captured per document including API round-trip time where applicable.

Error decomposition breaks down character-level edit operations into substitutions (characters recognized incorrectly), insertions (characters in the prediction that should not be there — hallucinated content), and deletions (characters in the ground truth missing from the prediction). These are computed via `rapidfuzz` edit operations and reported both as absolute counts and as rates normalized by ground truth length. This decomposition provides diagnostic insight into whether a model's errors are primarily recognition failures (substitution-dominant) or text generation artifacts (insertion-dominant).

CER and WER values can exceed 1.0 when the prediction is substantially longer than the ground truth — a pattern observed in receipt documents where OCR systems generate text from noisy backgrounds. This behavior is noted explicitly in the results rather than capped, as it provides diagnostic information about model failure modes.

### 4.4 Reporting

The `evaluate.py` script generates HTML reports and optional CSV exports from completed batch runs. Reports include per-model summary statistics, per-document breakdowns, and cross-model comparison tables suitable for stakeholder presentation.

Non-automated dimensions such as deployment complexity, privacy posture, and integration effort are discussed qualitatively in this whitepaper rather than computed by the framework. Extending the framework to capture these dimensions programmatically is a Phase 2 consideration.

## 5. Models Included in Phase 1

### 5.1 Tesseract — Local Baseline

Tesseract serves as the reference baseline for this evaluation. It is an open-source OCR engine maintained by Google, runs entirely on CPU, and requires no API credentials or cloud connectivity. The Phase 1 configuration uses a multi-language pack (`eng+hin+tel+tam+ben`) which, while enabling multilingual recognition, introduces noise on English-only documents where the additional language models generate spurious character hypotheses.

Tesseract's role in this benchmark is to establish a lower bound against which more sophisticated systems can be measured. It is not positioned as a production recommendation for complex enterprise documents.

### 5.2 Mistral OCR — Cloud Vision-Language Model

Mistral OCR is accessed via the Mistral AI cloud API and represents the vision-language model class of OCR systems. It processes documents by combining visual understanding with language modeling, which enables it to handle diverse document types without document-type-specific configuration. It achieved 100% success rate across all 91 documents in the full-dataset evaluation, including all 30 handwritten Devanagari character images where Tesseract failed entirely, demonstrating robust generalization across document categories. Its output is markdown-formatted, which the normalization pipeline strips before metric computation.

### 5.3 Docling — Structure-Aware Document Conversion

Docling is an open-source document conversion library that prioritizes preserving document structure in its output, producing markdown-style formatting that retains heading hierarchies, table structures, and layout relationships. This structural awareness comes at a latency cost — Docling's processing time on the sanity-check sample was approximately 60 seconds, significantly higher than all other evaluated systems.

During batch evaluation, the integration surfaced a serialization incompatibility where the `num_pages` attribute in the installed Docling version resolved to a method object rather than an integer, requiring a normalization fix in the wrapper.

### 5.4 PaddleOCR — High-Performance Open-Source Engine

PaddleOCR is developed by Baidu and represents the high-performance open-source OCR category. It runs locally without cloud dependencies and delivered the lowest CER among all models on the common sanity-check sample (0.3058), with strong F1 performance (0.7727).

The integration required significant wrapper adaptation. The existing wrapper targeted an older PaddleOCR API version and used incompatible constructor arguments (`show_log`, `use_gpu`) and runtime call patterns (`ocr(..., cls=True)`). The fix involved dynamic detection of supported constructor arguments, migration to the `predict()` API with `rec_texts`/`rec_scores` extraction on newer versions, and retention of fallback support for the legacy `ocr()` output shape.

### 5.5 Surya — Multilingual OCR Toolkit

Surya is an open-source multilingual OCR system that targets diverse script support. On the common sanity-check sample, it achieved competitive accuracy (CER 0.3388, F1 0.7407) at the cost of higher latency (13,437 ms), the second-highest among evaluated systems after Docling.

The wrapper required complete modernization. The original implementation targeted an older Surya API (`surya.ocr` module with legacy model/processor imports) that no longer exists in Surya 0.17.x. The fix migrated to the current predictor architecture using `FoundationPredictor`, `DetectionPredictor`, and `RecognitionPredictor` with the `TaskNames.ocr_with_boxes` interface.

### 5.6 Sarvam Vision OCR — India-Focused Document Intelligence

Sarvam Vision OCR is a cloud-hosted document intelligence service focused on Indian languages and enterprise document types. It achieved the highest token-level F1 score on the common sanity-check sample (0.7794) and the lowest WER (0.5060) among all six models, suggesting strong token-level extraction quality.

The integration was the most complex of any Phase 1 model. The original wrapper assumed a nonexistent one-shot `/v1/ocr` endpoint; the actual Sarvam API uses an asynchronous Document Intelligence job workflow. The rewrite used the `sarvamai` SDK to implement a five-step process: create job, upload file, start job, poll for completion, download ZIP output, and extract markdown text from the archive. Additionally, empty YAML configuration sections in the framework's config loader caused environment variable overlay failures, requiring a defensive fix in `utils/helpers.py`.

### 5.7 Models Planned for Phase 2

The framework also contains registered wrappers for DeepSeek OCR, Qwen-VL, OlmOCR, and GOT-OCR. These open-source vision-language models were not benchmark-ready for Phase 1. GOT-OCR completed a partial run of 21 ground-truth documents but produced unusable output with an average CER of 4.76 and a 92.3% insertion rate — the model generates multiple passes of document text plus hallucinated content on commercial documents, yielding outputs three to nine times longer than the ground truth. Its results appear only in the Phase 2 diagnostic table and not in recommendations. OlmOCR's run produced zero raw outputs in the current environment, indicating a load or runtime failure that remains unresolved. Qwen-VL produced 17 raw outputs before stalling, likely due to GPU memory or timeout constraints. DeepSeek OCR has a registered wrapper but has not yet been executed. Full resolution of these Phase 2 integrations is discussed in Section 13.

## 6. Sanity-Check Findings on Common Sample

All six Phase 1 models were validated against a common test document (`test-dataset/02_complex_tables/forms/0012199830.png`) to verify correct integration and establish single-document baseline measurements. These results are derived from a single document and should not be used for cross-model ranking; they serve only as integration validation and as an illustrative comparison point.

| Model | Latency (ms) | CER | WER | F1 |
|:---|---:|---:|---:|---:|
| tesseract | 2,896 | 0.5289 | 0.8072 | 0.4030 |
| mistral_ocr | 1,516 | 0.3347 | 0.5422 | 0.7313 |
| docling | 4,359 | 0.4236 | 0.7831 | 0.7059 |
| paddleocr | 6,686 | 0.3058 | 0.5301 | 0.7727 |
| surya | 13,437 | 0.3388 | 0.5542 | 0.7407 |
| sarvam_ocr | 3,677 | 0.3430 | 0.5060 | 0.7794 |

On this single structured form, the five non-baseline models clustered within a CER range of 0.3058 to 0.4236, all substantially outperforming Tesseract's 0.5289. PaddleOCR achieved the lowest CER while Sarvam Vision OCR achieved the highest F1 and lowest WER, suggesting that the two systems optimize for different aspects of extraction quality. Docling's higher CER relative to its F1 reflects its tendency to include structural formatting in output that affects character-level alignment while preserving token-level content. Mistral OCR delivered competitive accuracy at the lowest latency among the non-baseline models (1,516 ms), highlighting the efficiency advantages of cloud-hosted vision-language architectures.

These observations are provisional and drawn from a single document. The full-dataset results in Section 7 provide the appropriate evidence base for comparative analysis.

## 7. Full-Dataset Benchmark Results

### 7.1 Execution Summary

The full-dataset evaluation processed 91 documents across all four categories for Tesseract, Mistral OCR, Surya, and Sarvam Vision OCR. Docling completed a subsequent batch run over the complex-tables and mixed-content categories (42 of 45 GT documents). PaddleOCR completed only the financial subcategory (7 documents) before timeouts terminated the run. The difference between the 101-document corpus and the 91 documents processed by the initial batch reflects 10 documents that were excluded due to format incompatibilities or path resolution issues during batch execution.

Ground truth annotations are available for 45 documents across five subcategories: forms (n=5, human-verified), financial (n=9), multi-column (n=11), equations/formulas (n=10), and receipts (n=10). The 40 non-forms annotations are model-consensus GT generated by `generate_ground_truth.py` using Mistral OCR's normalized output as the primary candidate cross-validated against other models. This produces an unbiased GT for forms only; on non-forms categories, Mistral is excluded from cross-model ranking to avoid circularity. The original receipt ground truth files contained structured JSON with four extracted fields rather than full OCR text, making character- and word-level metrics against them meaningless; these were superseded by the consensus-GT procedure. Handwritten Devanagari characters and the single Indian-language document remain without metric-evaluable GT.

All quantitative metrics reported in this section declare the GT tier used. Cross-model claims on consensus GT are qualified as such and exclude Mistral. Findings on forms (n=5) represent the only unbiased comparison — the small sample size bounds the strength of statistical claims on that subset.

### 7.2 Tesseract — Full-Dataset Results

Tesseract processed 60 of 91 documents successfully, yielding a 65.9% success rate. The failures were concentrated entirely in the handwritten Devanagari character category (0 of 30 successful) and partially in equations/formulas (14 of 15). The complete failure on Devanagari characters is expected — these are 32x32 pixel isolated character images that fall below the minimum resolution and contextual information that Tesseract's recognition pipeline requires, despite including the Hindi language pack.

| Metric | Forms (n=5) | Receipts (n=10) | All GT (n=15) |
|:---|---:|---:|---:|
| Avg CER | 0.4651 | 0.8340 | 0.7110 |
| Avg WER | 0.6489 | 1.2631 | 1.0584 |
| Avg F1 | 0.4980 | 0.5362 | 0.5234 |
| Avg Precision | 0.5469 | 0.4430 | 0.4776 |
| Avg Recall | 0.4637 | 0.7540 | 0.6572 |
| Avg Word Accuracy | — | — | 0.2490 |
| Avg Edit Distance | — | — | 0.4391 |

Tesseract's receipt CER (0.8340) exceeds 1.0 on several individual documents, indicating that the model generates more characters than exist in the ground truth. The precision-recall split on receipts (0.4430 precision, 0.7540 recall) confirms that Tesseract over-generates text, extracting most ground truth tokens but also producing substantial hallucinated content from backgrounds, logos, and visual noise. On forms, the more balanced precision-recall profile (0.5469/0.4637) suggests recognition errors rather than hallucination.

**Category-level success rates:**

| Category | Success Rate | Avg Latency (ms) |
|:---|---:|---:|
| 02_complex_tables/financial | 15/15 (100%) | 10,611 |
| 02_complex_tables/forms | 5/5 (100%) | 3,232 |
| 02_complex_tables/multi_column | 15/15 (100%) | 12,855 |
| 03_handwritten/hindi_devanagari | 0/30 (0%) | — |
| 04_indian_languages/hindi | 1/1 (100%) | 1,011 |
| 06_mixed_content/equations_formulas | 14/15 (93.3%) | 17,581 |
| 06_mixed_content/receipts | 10/10 (100%) | 3,849 |

### 7.3 Mistral OCR — Full-Dataset Results

Mistral OCR achieved a 100% success rate across all 91 documents, the only model in the initial comparison to process every document without failure. This includes all 30 handwritten Devanagari character images where Tesseract failed completely, demonstrating the robustness advantage of vision-language model architectures even on minimal-context character-level inputs.

| Metric | Forms (n=5) | Receipts (n=10) | All GT (n=15) |
|:---|---:|---:|---:|
| Avg CER | 0.3933 | 0.0000* | 0.1311 |
| Avg WER | 0.5033 | 0.0000* | 0.1678 |
| Avg F1 | 0.7549 | 1.0000* | 0.9183 |
| Avg Precision | 0.7582 | 1.0000* | 0.9194 |
| Avg Recall | 0.7584 | 1.0000* | 0.9195 |
| Avg Word Accuracy | — | — | 0.8322 |
| Avg Edit Distance | — | — | 0.1161 |

*Receipt metrics are artificially perfect (CER=0.0, F1=1.0) because the receipt ground truth was derived from Mistral OCR's own normalized output. These values should be excluded from cross-model comparison. On forms alone, Mistral achieves competitive but not dominant accuracy (F1 0.7549, CER 0.3933).

**Category-level success rates:**

| Category | Success Rate | Avg Latency (ms) |
|:---|---:|---:|
| 02_complex_tables/financial | 15/15 (100%) | 30,172 |
| 02_complex_tables/forms | 5/5 (100%) | 5,292 |
| 02_complex_tables/multi_column | 15/15 (100%) | 61,296 |
| 03_handwritten/hindi_devanagari | 30/30 (100%) | 1,945 |
| 04_indian_languages/hindi | 1/1 (100%) | 5,812 |
| 06_mixed_content/equations_formulas | 15/15 (100%) | 5,111 |
| 06_mixed_content/receipts | 10/10 (100%) | 3,169 |

### 7.4 Surya — Full-Dataset Results

Surya processed 71 of 91 documents successfully, yielding a 78.0% success rate. The failures were concentrated in the handwritten Devanagari character category (10 of 30 successful) where Surya's recognition engine produced output for some character images but not others. Surya's latency was the highest among all evaluated models, averaging 52,814 ms per document, with multi-column and equations documents exceeding 80,000 ms on average.

| Metric | Forms (n=5) | Receipts (n=10) | All GT (n=15) |
|:---|---:|---:|---:|
| Avg CER | 0.3028 | 0.7575 | 0.6059 |
| Avg WER | 0.4483 | 0.9917 | 0.8105 |
| Avg F1 | 0.7710 | 0.6704 | 0.7039 |
| Avg Precision | 0.7943 | 0.5629 | 0.6400 |
| Avg Recall | 0.7508 | 0.8994 | 0.8499 |
| Avg Word Accuracy | — | — | 0.3793 |
| Avg Edit Distance | — | — | 0.3574 |

Surya achieved the lowest CER on forms (0.3028) of any evaluated model, confirming its competitive character-level fidelity on structured documents. On receipts, Surya's precision-recall split (0.5629/0.8994) reveals the same over-generation pattern seen in Tesseract — high recall with low precision, indicating hallucinated content from noisy receipt backgrounds. Despite its high latency, Surya's accuracy profile makes it a viable option for batch processing workflows where throughput is not the primary constraint.

**Category-level success rates:**

| Category | Success Rate | Avg Latency (ms) |
|:---|---:|---:|
| 02_complex_tables/financial | 15/15 (100%) | 31,811 |
| 02_complex_tables/forms | 5/5 (100%) | 38,289 |
| 02_complex_tables/multi_column | 15/15 (100%) | 80,326 |
| 03_handwritten/hindi_devanagari | 10/30 (33.3%) | 2,716 |
| 04_indian_languages/hindi | 1/1 (100%) | 8,220 |
| 06_mixed_content/equations_formulas | 15/15 (100%) | 95,815 |
| 06_mixed_content/receipts | 10/10 (100%) | 40,371 |

### 7.5 Sarvam Vision OCR — Full-Dataset Results

Sarvam Vision OCR processed 62 of 91 documents successfully, yielding a 68.1% success rate. The failures reveal a distinctive pattern: Sarvam failed on all 10 receipt documents and 13 of 15 equations/formulas documents, but succeeded on all 45 complex table documents and 24 of 30 handwritten Devanagari character images. This profile suggests that Sarvam's Document Intelligence service is optimized for structured and form-based documents rather than noisy or visually complex content.

| Metric | Forms (n=5) | All GT (n=5) |
|:---|---:|---:|
| Avg CER | 0.5124 | 0.5124 |
| Avg WER | 0.5183 | 0.5183 |
| Avg F1 | 0.7592 | 0.7592 |
| Avg Precision | 0.7479 | 0.7479 |
| Avg Recall | 0.7788 | 0.7788 |
| Avg Word Accuracy | — | 0.4817 |
| Avg Edit Distance | — | 0.3794 |

Sarvam's metrics are computed on only 5 ground-truth documents (all forms), as it failed on all receipt documents where the remaining 10 ground-truth annotations exist. On forms, Sarvam achieved competitive F1 (0.7592) with a relatively balanced precision-recall profile (0.7479/0.7788). Its CER (0.5124) is elevated relative to Surya and Mistral, suggesting more verbose output that captures content tokens while including additional formatting. Its latency (4,688 ms average) is the lowest among all non-baseline models, reflecting efficient cloud API design despite the asynchronous job workflow.

**Category-level success rates:**

| Category | Success Rate | Avg Latency (ms) |
|:---|---:|---:|
| 02_complex_tables/financial | 15/15 (100%) | 5,106 |
| 02_complex_tables/forms | 5/5 (100%) | 3,836 |
| 02_complex_tables/multi_column | 15/15 (100%) | 6,597 |
| 03_handwritten/hindi_devanagari | 24/30 (80.0%) | 3,481 |
| 04_indian_languages/hindi | 1/1 (100%) | 4,067 |
| 06_mixed_content/equations_formulas | 2/15 (13.3%) | 4,165 |
| 06_mixed_content/receipts | 0/10 (0%) | — |

### 7.6 Docling — Full-Dataset Results

Docling completed its batch run over the complex-tables and mixed-content categories, processing 42 of 45 ground-truth documents. The three absent documents (two multi-column, one receipt) failed individual processing and did not produce outputs; this is reflected in its reduced n values per subcategory. Docling's average processing latency exceeds 60 seconds per document on the evaluation hardware, with multi-column and receipt categories approaching 70-75 seconds per document on average.

| Metric | Forms (n=5) | Financial (n=9) | Multi-col (n=9) | Equations (n=10) | Receipts (n=9) | Consensus GT all (n=37) |
|:---|---:|---:|---:|---:|---:|---:|
| Avg CER | 0.4302 | 0.3541 | 0.3707 | 0.4271 | 0.6167 | 0.4417 |
| Avg WER | 0.5593 | 0.3996 | 0.4091 | 0.5384 | 0.7751 | 0.5308 |
| Avg F1 | 0.5889 | 0.7003 | 0.7827 | 0.7587 | 0.6729 | 0.7294 |
| Avg Precision | 0.8255 | 0.7743 | 0.8556 | 0.8439 | 0.7584 | 0.8090 |
| Avg Recall | 0.5586 | 0.6532 | 0.7386 | 0.7415 | 0.7023 | 0.7098 |

Docling's precision-recall split (0.8090/0.7098 on consensus GT) reveals a conservative extraction pattern: the text it produces is generally correct, but it tends to omit content. This aligns with its error decomposition, which shows 63.4% of errors are deletions — by far the highest deletion fraction among all evaluated models. Operationally this is a safer profile than insertion-dominant behavior because missing content is easier to detect and recover than hallucinated content to filter. On multi-column layouts Docling delivered competitive F1 (0.7827), second only to Surya on the consensus subset, with the lowest WER (0.4091) among all evaluated models on that category.

### 7.6a PaddleOCR — Partial Results (Financial Only)

PaddleOCR encountered processing timeouts on non-financial categories during batch execution, with individual document processing times reaching 500-5,000 seconds. The run completed 7 of 9 financial ground-truth documents before the timeout gate terminated execution on subsequent categories. Reported metrics are limited to this partial evidence.

| Metric | Financial (n=7) |
|:---|---:|
| Avg CER | 0.4653 |
| Avg WER | 0.5885 |
| Avg F1 | 0.6846 |
| Avg Precision | 0.7291 |
| Avg Recall | 0.6509 |

PaddleOCR's partial financial F1 (0.6846) trails Surya (0.8081), Sarvam (0.7667), and Docling (0.7003) on the same category. Its error decomposition (28.8% substitutions, 11.6% insertions, 59.6% deletions) is deletion-dominant, similar to Docling. The earlier sanity-check result (F1 0.7727 on a single form) is not contradicted by these financial numbers but covers a different category and cannot be compared directly. Full PaddleOCR evaluation across forms, multi-column, equations, and receipts remains incomplete and is a Phase 2 carryover.

### 7.7 Cross-Model Comparison — Forms Subcategory (Human-Verified GT, n=5)

The forms subcategory is the only Phase 1 comparison backed by human-verified ground truth. All five fully-evaluated models have metrics here and Mistral is included without circularity.

| Model | Avg CER | Avg WER | Avg F1 | Avg Precision | Avg Recall |
|:---|---:|---:|---:|---:|---:|
| tesseract | 0.4651 | 0.6489 | 0.4980 | 0.5469 | 0.4637 |
| mistral_ocr | 0.3933 | 0.5033 | 0.7549 | 0.7582 | 0.7584 |
| surya | 0.3028 | 0.4483 | 0.7710 | 0.7943 | 0.7508 |
| sarvam_ocr | 0.5124 | 0.5183 | 0.7592 | 0.7479 | 0.7788 |
| docling | 0.4302 | 0.5593 | 0.5889 | 0.8255 | 0.5586 |

Surya achieved the lowest CER (0.3028) and the highest F1 (0.7710), suggesting the most faithful character-level and token-level reproduction. Mistral (F1 0.7549) and Sarvam (F1 0.7592) cluster within 0.02 F1 of Surya with differently-shaped precision-recall profiles: Mistral balanced (0.7582/0.7584), Sarvam recall-favoring (0.7479/0.7788). Docling's forms F1 (0.5889) trails the leading cluster but reflects a high-precision conservative profile (0.8255 precision, 0.5586 recall) — when Docling extracts text, it is accurate; it simply extracts less. Tesseract trails all four on every metric, confirming its position as a baseline reference rather than a production option.

At n=5, no pairwise F1 differences between the Mistral / Surya / Sarvam cluster reach statistical significance (Wilcoxon signed-rank test, all p>0.05). Statistical significance is available on the broader consensus-GT subset (Section 7.9). Expanding human-verified forms GT remains the highest-priority methodological improvement.

### 7.8 Cross-Model Comparison — Receipts Subcategory (Consensus GT, Mistral Excluded)

The receipts subcategory (n=10 consensus GT; Docling n=9 due to one processing failure) allows comparison between Tesseract, Surya, and Docling. Sarvam Vision OCR failed on all receipt documents and is excluded. Mistral OCR is excluded due to consensus-GT circularity — its own output contributed the primary candidate text used to build the receipt GT. Independent manually transcribed receipt ground truth would be needed to evaluate Mistral fairly on this category.

| Model | n | Avg CER | Avg WER | Avg F1 | Avg Precision | Avg Recall |
|:---|---:|---:|---:|---:|---:|---:|
| tesseract | 10 | 0.8340 | 1.2631 | 0.5362 | 0.4430 | 0.7540 |
| surya | 10 | 0.7575 | 0.9917 | 0.6704 | 0.5629 | 0.8994 |
| docling | 9 | 0.6167 | 0.7751 | 0.6729 | 0.7584 | 0.7023 |

Surya and Docling achieve nearly identical aggregate F1 on receipts (0.6704 vs 0.6729) but with opposing precision-recall profiles. Surya shows the characteristic over-generation pattern — high recall (0.8994) with low precision (0.5629), consistent with its insertion-heavy error profile. Docling is precision-favoring (0.7584/0.7023), consistent with its deletion-dominant error profile: it under-extracts receipt content but what it does extract is generally correct. The choice between them for a production receipt pipeline depends on downstream tolerance: a pipeline that can filter spurious insertions should prefer Surya; a pipeline that can reconcile missing fields from structured templates should prefer Docling. Tesseract trails both models on all metrics and shows the most aggressive over-generation (recall 0.7540, precision 0.4430).

These receipt metrics reflect genuine OCR quality differences but should be read with the caveat that the ground truth was machine-generated rather than manually transcribed. Any systematic biases in Mistral's receipt extraction would propagate to the consensus GT and affect the relative scoring of other models.

### 7.9 Cross-Model Comparison — Consensus GT Categories (Mistral Excluded)

Beyond forms and receipts, three additional subcategories (financial, multi-column, equations/formulas) now have consensus ground truth enabling expanded cross-model comparison. Mistral is excluded. Aggregate consensus-GT F1 by model across all non-forms subcategories:

| Model | n | Avg CER | Avg WER | Avg F1 | Avg Precision | Avg Recall |
|:---|---:|---:|---:|---:|---:|---:|
| surya | 40 | 0.5673 | 0.7707 | 0.7717 | 0.7275 | 0.8650 |
| docling | 37 | 0.4417 | 0.5308 | 0.7294 | 0.8090 | 0.7098 |
| sarvam_ocr | 22 | 0.6544 | 0.7859 | 0.7290 | 0.6801 | 0.8524 |
| paddleocr | 7 | 0.4653 | 0.5885 | 0.6846 | 0.7291 | 0.6509 |
| tesseract | 40 | 0.5110 | 0.7585 | 0.6221 | 0.5904 | 0.7087 |

Three observations emerge from the expanded comparison. First, Surya leads on aggregate F1 (0.7717) across all consensus categories, with the highest recall (0.8650) reflecting its tendency to extract most content at the cost of over-generation. Second, Docling's aggregate F1 (0.7294) is competitive with Sarvam's (0.7290), but achieved through very different trade-offs — Docling has the highest precision (0.8090) of any evaluated model on the consensus subset, while Sarvam has the highest recall after Surya (0.8524). Third, Tesseract trails all non-baseline models on consensus F1 by 0.10 or more, consistent with its sanity-check and forms-subcategory results.

Pairwise Wilcoxon signed-rank tests on the consensus subset yield four statistically significant F1 comparisons at p<0.05: Surya outperforms Tesseract (n=45, p<0.001, large effect d=0.850), Docling outperforms Tesseract (n=42, p<0.001, medium effect d=0.576), Surya outperforms Docling (n=42, p=0.024, small effect d=0.314), and all three significantly outperform GOT-OCR. Surya versus Sarvam trends in Surya's favor but does not reach significance (n=27, p=0.091). Tesseract versus Sarvam does not reach significance (n=27, p=0.148). These are the first statistically supported pairwise comparisons in this benchmark; on human-verified forms alone (n=5), no pairwise test reaches significance, underscoring the value of the expanded corpus while also acknowledging that significance on consensus GT partly reflects inter-model agreement rather than agreement with human truth.

## 8. Error Decomposition Analysis

Error decomposition provides diagnostic insight beyond aggregate CER by revealing the structural composition of each model's errors. The substitution-insertion-deletion (S/I/D) profile characterizes whether a model's errors arise primarily from misrecognition (substitutions), hallucination of content not present in the source document (insertions), or failure to capture content that exists in the source (deletions).

### 8.1 Error Profiles by Model

| Model | n | Total Errors | Substitutions | Insertions | Deletions | Dominant Mode |
|:---|---:|---:|---:|---:|---:|:---|
| mistral_ocr | 45 | 1,615 | 549 (34.0%) | 604 (37.4%) | 462 (28.6%) | Balanced |
| surya | 45 | 53,740 | 23,682 (44.1%) | 19,096 (35.5%) | 10,962 (20.4%) | Substitution-heavy |
| tesseract | 45 | 29,589 | 7,823 (26.4%) | 10,760 (36.4%) | 11,006 (37.2%) | Deletion-dominant |
| docling | 42 | 29,471 | 5,303 (18.0%) | 5,480 (18.6%) | 18,688 (63.4%) | Deletion-dominant |
| sarvam_ocr | 27 | 21,977 | 1,927 (8.8%) | 17,852 (81.2%) | 2,198 (10.0%) | Insertion-dominant |
| paddleocr | 7 | 7,449 | 2,147 (28.8%) | 865 (11.6%) | 4,437 (59.6%) | Deletion-dominant |
| got_ocr | 21 | 104,117 | 6,655 (6.4%) | 96,060 (92.3%) | 1,402 (1.3%) | Hallucination-dominant |

Four distinct error profiles emerge from the expanded decomposition. Mistral OCR remains the most balanced model with the lowest total error count across its 45-document subset (1,615 errors, roughly 36 per document); its composition is nearly uniform across S/I/D. Surya's profile on the expanded corpus is substitution-heavy (44.1%) with moderate insertions (35.5%), a different profile than the original 15-doc subset indicated — the broader consensus subset reveals that Surya makes character-level recognition errors at scale rather than exclusively over-generating. Docling and PaddleOCR exhibit strongly deletion-dominant profiles (63.4% and 59.6% deletions respectively), consistent with their high precision and lower recall — these models under-extract rather than hallucinate. Sarvam's expanded profile is extreme insertion-dominant (81.2%), reinforcing that its verbose output pattern generalizes beyond the forms subset. GOT-OCR's 92.3% insertion rate alongside a CER exceeding 4.0 indicates a qualitative failure mode — it generates multiple passes of document text plus invented content — and is reported here for diagnostic completeness rather than comparative ranking.

### 8.2 Diagnostic Implications

The expanded decomposition reveals that post-processing strategy should be model-specific. Insertion-dominant models (Sarvam, GOT-OCR) benefit from length-ratio checks or confidence-gated filtering that flag predictions substantially longer than expected. Deletion-dominant models (Docling, PaddleOCR, Tesseract on receipts) benefit from recall-boosting interventions: multi-pass processing, region-of-interest expansion, or supplementary line-detection. Substitution-heavy models (Surya) benefit from character-level confidence models or language-model rescoring. Mistral's balanced profile suggests a mature model whose errors are approximately evenly distributed — the largest improvement path for Mistral-based pipelines is error reduction generally rather than any specific class of post-processing.

## 9. Confidence Calibration

Model confidence scores are valuable for production pipelines only if they correlate with actual extraction accuracy. A well-calibrated model allows downstream systems to flag low-confidence extractions for human review, reducing error propagation.

Phase 1 evaluated confidence calibration for the two models that report document-level confidence scores: Tesseract and Surya. Spearman rank correlation between reported confidence and document-level F1 was computed across all ground-truth documents.

Tesseract's confidence scores show strong positive correlation with F1 (Spearman rho=+0.83, p=0.0001), indicating that its confidence estimates are meaningful for quality gating — low-confidence Tesseract outputs reliably predict poor extraction quality and can be routed for human review. Surya's confidence scores show no significant correlation with F1 (Spearman rho=+0.08, p>0.05), meaning its confidence values should not be used for quality-based routing without additional calibration work.

This finding has practical architectural implications: Tesseract's confidence, despite its lower overall accuracy, provides a reliable signal for human-in-the-loop pipelines. Surya's higher accuracy is not accompanied by self-awareness of its own performance, requiring external quality estimation mechanisms if used in workflows that need to identify problematic extractions.

## 10. Integration and Engineering Observations

One of the distinctive contributions of this evaluation is that it documents the real integration complexity encountered when operationalizing OCR systems, not just their accuracy once running. The following observations emerged during Phase 1 wrapper development and provide actionable intelligence for enterprise teams evaluating OCR adoption.

### 10.1 API Stability and Version Drift

Three of the six Phase 1 models (PaddleOCR, Surya, and Sarvam Vision) required significant wrapper rewrites due to API changes between the version assumed by the original wrapper code and the version actually installed. PaddleOCR changed its constructor signature and runtime API between major versions. Surya completely restructured its module hierarchy and predictor interface in version 0.17.x. Sarvam Vision's actual API workflow bore no resemblance to the endpoint assumed by the original wrapper.

This pattern suggests that OCR library APIs are changing rapidly and that enterprise integration teams should budget for ongoing wrapper maintenance. Pinning dependency versions in production is necessary but insufficient — the wrapper itself must be version-aware.

### 10.2 Asynchronous and Job-Based Cloud APIs

The Sarvam Vision integration revealed that some cloud document intelligence services use asynchronous job workflows rather than synchronous request-response patterns. The actual API requires creating a job, uploading the file, starting the job, polling for completion, downloading a ZIP archive, and extracting the text output. This five-step workflow has latency and error-handling implications that are invisible in benchmark accuracy comparisons but critical for production pipeline design.

### 10.3 Serialization and Batch Pipeline Robustness

Docling's batch evaluation surfaced a serialization failure where a metadata attribute resolved to a callable object rather than a primitive value, crashing the JSON export pipeline. This class of bug is invisible in single-document testing and only manifests at batch scale. It underscores the importance of end-to-end batch validation before trusting a model integration for production use.

### 10.4 Configuration and Environment Management

The framework's YAML-based configuration system required a defensive fix when empty configuration sections caused `None` values to propagate into environment variable overlay logic. This is a common pattern in multi-model configurations where not all models require all configuration sections, and it highlights the need for null-safe configuration handling in evaluation frameworks that must support heterogeneous model requirements.

### 10.5 Summary of Integration Lessons

| Model | Initial Problem | Fix Applied | Operational Takeaway |
|:---|:---|:---|:---|
| PaddleOCR | Outdated constructor and API | Dynamic arg detection, `predict()` migration | Library version drift requires version-aware wrappers |
| Surya | Deprecated module structure | Full predictor architecture migration | Upstream API movement can break integrations silently |
| Sarvam OCR | Wrong endpoint assumption | Five-step SDK job workflow | Cloud APIs may be asynchronous and job-based |
| Docling | Non-serializable metadata | Metadata normalization | Batch pipelines need strict JSON-safe outputs |

These integration challenges are not incidental to the benchmark — they are findings in their own right, and should inform enterprise evaluation criteria alongside pure accuracy metrics.

## 11. Recommendations

The following recommendations are based on Phase 1 full-dataset evidence from five models (Tesseract, Mistral OCR, Surya, Sarvam Vision OCR, and Docling) and partial financial-only evidence for PaddleOCR. They are organized by use case to reflect the reality that no single OCR system dominates across all document types and operational constraints. All recommendations should be read as provisional given the small human-verified forms sample (n=5); statistical significance is available on the expanded 45-document consensus GT but with the documented caveat that consensus-GT agreement partly reflects inter-model alignment rather than alignment with ground truth.

### 11.1 Structured Forms and Tables

For structured forms, Phase 1 evidence suggests a competitive tier of three models. Surya achieved the lowest CER (0.3028) and highest F1 (0.7710) on forms, suggesting the strongest character-level and token-level extraction quality, at the cost of significantly higher latency (38,289 ms average on forms). Mistral OCR provides the strongest combination of accuracy and reliability, with competitive forms F1 (0.7549), a 100% success rate across all document categories, and moderate latency. Sarvam Vision OCR offers low latency (3,836 ms on forms) with competitive F1 (0.7592) and 100% success on all table categories, making it the strongest option for latency-sensitive form processing.

For enterprise teams requiring a single recommendation: Mistral OCR offers the best balance of accuracy, reliability, and generalization across diverse document types. For teams with form-focused workloads and latency constraints, Sarvam Vision OCR merits strong consideration.

### 11.2 Handwritten Character Recognition

The handwritten Devanagari category in this evaluation consists of 32x32 pixel isolated character images from the DHCD dataset. These test character-level recognition under minimal-context conditions rather than full-document handwriting recognition. With this caveat, the category differentiates models sharply: Tesseract failed entirely (0/30), Surya succeeded partially (10/30), Sarvam succeeded on most (24/30), and Mistral OCR processed all 30 successfully. Vision-language model architectures appear more robust on these minimal-context inputs, though conclusions about full-document handwriting recognition would require a different test corpus with page-scale handwritten documents.

### 11.3 Receipt Processing

Receipt processing shows competitive accuracy from Surya (F1 0.6704) and Docling (F1 0.6729) on consensus GT, with opposing precision-recall profiles. Surya is recall-favoring (0.8994 recall, 0.5629 precision) — it captures most receipt content but hallucinates substantial spurious text from backgrounds, logos, and barcodes. Docling is precision-favoring (0.7584 precision, 0.7023 recall) — it under-extracts but what it produces is generally correct. Tesseract trails both (F1 0.5362) with aggressive over-generation. Sarvam Vision OCR failed on all receipt documents. Mistral OCR cannot be evaluated fairly on receipts due to consensus-GT circularity.

The Surya/Docling choice depends on downstream pipeline architecture: pipelines with confidence-gated filtering or length-ratio checks should prefer Surya (to recover the recall advantage); pipelines built around structured field extraction should prefer Docling (to avoid hallucinated fields). Dedicated receipt OCR models or field-extraction systems remain a viable alternative for production use cases where generic OCR quality is insufficient.

### 11.4 Structure-Preserving Extraction

For use cases requiring preserved document structure (headings, tables, layout), Docling completed its first full batch run in Phase 1 and remains the only evaluated system that natively produces structured markdown output. Its consensus-GT F1 of 0.7294 (n=37) and human-verified forms F1 of 0.5889 place it in the competitive mid-tier for plain-text extraction, with a deletion-dominant error profile (63.4%) that is operationally safer than insertion-dominant alternatives. The latency penalty — averaging 60-75 seconds per document on non-trivial categories — must be weighed against the downstream value of preserved structure in document conversion pipelines. For batch workflows where throughput is not the primary constraint, Docling is the recommended structure-preserving option.

### 11.5 Baseline and Reference Testing

Tesseract remains appropriate as a cost-free, dependency-light baseline for testing pipeline infrastructure and establishing lower-bound expectations. Its confidence scores are well-calibrated (rho=+0.83 with F1), making it useful as a quality-gating mechanism in human-in-the-loop workflows despite its lower overall accuracy. It should not be used as a production OCR engine for complex enterprise documents given its 65.9% success rate, complete failure on character-level inputs, weakest accuracy on all metric categories, and insertion-heavy error profile on receipts.

## 12. Limitations

This Phase 1 evaluation has several important limitations that bound the strength of its conclusions.

The human-verified ground-truth subset is limited to 5 forms documents. The expanded 45-document ground-truth corpus adds 40 non-forms annotations generated by a model-consensus procedure that uses Mistral OCR's output as the primary candidate cross-validated against other models. This creates two distinct biases that must be acknowledged in every numerical claim. First, Mistral OCR cannot be fairly compared to other models on consensus categories because its own output contributed to the reference, and is excluded from all such cross-model rankings. Second, even with Mistral excluded, the consensus GT measures agreement with a specific multi-model ensemble rather than with independently verified truth. Statistical significance on the expanded subset therefore partly reflects inter-model agreement patterns rather than agreement with ground truth in the strict sense. The highest-priority methodological improvement for Phase 2 is expansion of human-verified annotations, particularly for financial, multi-column, equations, and receipt categories.

At the sample size of n=5 for human-verified forms, no pairwise F1 differences between Mistral, Surya, Sarvam, and Docling reach statistical significance (Wilcoxon signed-rank test, all p>0.05). Significance is available on the consensus subset (Surya > Tesseract, Docling > Tesseract, Surya > Docling, all at p<0.05) but with the caveat described above.

The handwritten category consists of 32x32 pixel isolated Devanagari character images from the DHCD dataset, not page-scale handwritten documents. Results on this category reflect character-level recognition capability under minimal-context conditions and should not be extrapolated to full-document handwriting recognition scenarios. The Indian languages category contains a single document, making any multilingual claim statistically unsupported.

Full-dataset results are available for five of six Phase 1 models (Tesseract, Mistral OCR, Surya, Sarvam Vision OCR, and Docling). PaddleOCR completed only the financial subcategory (7 documents) before timeouts terminated execution; its full-dataset evaluation across forms, multi-column, equations, and receipts remains a Phase 2 carryover. On Phase 2 models: GOT-OCR produced catastrophic hallucination (CER 4.76, 92.3% insertions) and is reported as a diagnostic only; OlmOCR produced zero raw outputs on the current environment; Qwen-VL stalled after 17 documents. These integrations require environment and runtime investigation before they can contribute meaningful benchmark evidence.

Success rates vary significantly by model: Mistral OCR achieved 100%, Surya 78.0%, Sarvam Vision OCR 68.1%, and Tesseract 65.9%. Models that failed on specific document categories produce aggregate metrics that are not directly comparable to models with full category coverage, as the failed categories are excluded from their metrics.

Latency comparisons between local models (Tesseract, PaddleOCR, Surya, Docling) and cloud-hosted models (Mistral OCR, Sarvam Vision OCR) are not directly comparable, as cloud latency includes network round-trip time and API overhead that varies with server load and geographic proximity. Local model latency depends on the evaluation machine's hardware configuration.

Several evaluation dimensions relevant to enterprise adoption — including cost per document, privacy and data residency compliance, deployment complexity, and structured extraction quality beyond plain text — are not automatically measured by the framework and are discussed only qualitatively in this paper.

## 13. Phase 2 Roadmap

Phase 2 of this evaluation addresses the limitations identified above and expands the benchmark into a more comprehensive enterprise OCR comparison.

Expanding human-verified ground truth is the single highest-priority methodological improvement. The current expanded 45-document corpus is dominated by model-consensus annotations and supports cross-model claims only with explicit exclusion of Mistral. A target of at least 30 human-verified documents per subcategory would support statistically significant pairwise comparisons without inter-model-agreement bias. Manually transcribed receipt ground truth would eliminate the Mistral circularity entirely. The Indian languages category requires at least 15-20 additional documents across Hindi, Tamil, Telugu, and Bengali before any multilingual claim can be substantiated. The handwritten category should be supplemented with page-scale handwritten documents rather than the existing 32x32 character crops. European-language documents and internal enterprise document samples would further strengthen the benchmark's relevance.

On the model side, Phase 2 must resolve three integration issues surfaced during Phase 1. GOT-OCR's hallucination failure mode (CER 4.76, 92.3% insertions) is not a benchmark-configuration problem — the model produces multiple passes of document text plus invented content on commercial layouts. Its inclusion in any comparison requires either a task-specific fine-tuning pass or a length-ratio post-filter that discards outputs exceeding plausible document length. OlmOCR's zero-output behavior on the current environment requires runtime investigation (likely model load, tokenizer mismatch, or GPU assignment). Qwen-VL's stall after 17 documents suggests GPU memory exhaustion or per-document timeout; a reduced-precision inference configuration or process-level isolation should resolve it. DeepSeek OCR has a registered wrapper and requires only a first execution pass. Completing PaddleOCR's full-dataset evaluation (currently financial-only) by raising per-document timeouts and partitioning the run by category is a straightforward carryover.

Extending the metrics framework to capture cost per document, structured extraction quality (table cell accuracy, layout fidelity), confidence calibration for additional models, and deployment complexity scores would move the evaluation closer to a comprehensive decision-support tool for enterprise architects.

## 14. Conclusion

This Phase 1 benchmark demonstrates that a practical, reproducible OCR evaluation framework can be built and operated to produce evidence-backed model comparisons across diverse enterprise document types. The framework's self-registering model architecture, automated metrics pipeline with text normalization and error decomposition, and batch evaluation engine provide a reusable foundation for ongoing OCR assessment as new models and document categories are added.

The integration process itself yielded findings that are as valuable as the accuracy metrics: rapid API evolution across open-source OCR libraries, the complexity of asynchronous cloud document intelligence workflows, and the importance of batch-scale validation for surfacing serialization and configuration failures that are invisible in single-document testing.

Phase 1 evidence suggests a clear stratification among OCR systems on structured forms (n=5 human-verified). Surya achieved the lowest CER (0.3028) and highest F1 (0.7710), demonstrating strong character-level fidelity at the cost of high latency. Mistral OCR provides the most robust cross-category coverage with a 100% success rate and competitive forms F1 (0.7549), making it the safest general-purpose option. Sarvam Vision OCR delivers competitive forms accuracy (F1 0.7592) at the lowest latency among non-baseline models (4,688 ms) but exhibits selective category support. Docling completed its first full batch run with consensus-GT F1 of 0.7294 and a deletion-dominant error profile — the only evaluated model that natively produces structured markdown output. Tesseract serves as a useful baseline with well-calibrated confidence scores (ρ=+0.83 with F1) but trails all models on accuracy metrics.

The expanded consensus-GT analysis yields the benchmark's first statistically significant pairwise comparisons: Surya outperforms Tesseract (p<0.001, large effect), Docling outperforms Tesseract (p<0.001, medium effect), and Surya outperforms Docling (p=0.024, small effect). These significances must be interpreted with the caveat that consensus GT partly reflects inter-model agreement.

Error decomposition across the expanded corpus reveals four distinct failure modes. Mistral is balanced. Surya is substitution-heavy (44.1%). Docling and PaddleOCR are deletion-dominant (63.4% and 59.6%). Sarvam is insertion-dominant (81.2%). GOT-OCR exhibits a qualitatively different hallucination failure mode (92.3% insertions, CER > 4.0) that disqualifies it from Phase 1 comparison. Each profile implies a different post-processing strategy for a production deployment.

These findings provide a defensible foundation for enterprise OCR selection decisions, with the explicit caveat that they are based on a 5-document human-verified subset and a 45-document consensus subset with documented biases. Phase 2 priorities are human-verified GT expansion, resolution of OlmOCR and Qwen-VL runtime failures, completion of PaddleOCR full-dataset evaluation, and DeepSeek OCR first execution.

## Appendix A: Environment and Reproducibility

**Runtime environment:** Python 3.12 with virtualenv-based dependency isolation. All model dependencies are specified in `requirements.txt` and locked for reproducibility.

**Hardware:** Evaluation conducted on a development workstation (macOS, Apple Silicon). Local model latencies reflect this specific hardware configuration.

**Key repository artifacts:**
- Framework code: `models/`, `utils/`, `configs/`
- Evaluation scripts: `run_single.py`, `run_model.py`, `run_batch.py`, `evaluate.py`
- Dataset: `test-dataset/` (101 documents)
- Results: `results/` (timestamped directories per run)
- This document: `docs/whitepaper/draft.md`
- Supporting analysis: `docs/whitepaper/tables.md`, `docs/whitepaper/sanity-checks-2026-04-09.md`
- Revival plan: `docs/plans/2026-04-09-ocr-whitepaper-revival-plan.md`

## Appendix B: Metric Definitions

| Metric | Formula | Interpretation |
|:---|:---|:---|
| CER | Levenshtein(pred, gt) / len(gt) | Lower is better; can exceed 1.0 if prediction is longer than ground truth |
| WER | Levenshtein(pred_words, gt_words) / len(gt_words) | Lower is better; word-level equivalent of CER |
| Word Accuracy | max(1 - WER, 0) | Higher is better; floored at 0 |
| Edit Distance | Levenshtein(pred, gt) / max(len(pred), len(gt)) | Lower is better; normalized to 0-1 range |
| Token F1 | 2 * P * R / (P + R), Counter-based | Higher is better; bag-of-words with multiplicity |
| Precision | TP / pred_total (Counter-based) | Higher is better; measures over-generation |
| Recall | TP / gt_total (Counter-based) | Higher is better; measures under-extraction |
| Substitution Rate | char_substitutions / len(gt) | Fraction of errors from misrecognition |
| Insertion Rate | char_insertions / len(gt) | Fraction of errors from hallucinated text |
| Deletion Rate | char_deletions / len(gt) | Fraction of errors from missing text |
| Latency | Wall-clock time including API round-trip | Measured in milliseconds per document |

## Appendix C: Text Normalization Pipeline

All OCR output and ground truth text is normalized before metric computation to ensure fair comparison across models with different output formats. The normalization pipeline applies the following steps in order:

1. **Unicode NFKC normalization** — canonicalizes equivalent Unicode representations, critical for Devanagari and accented characters where multiple encodings exist for visually identical text.
2. **Markdown/HTML stripping** — removes heading markers (`# `), bold/italic (`**`, `*`), inline code backticks, image/link syntax, table row/separator syntax, list bullets, numbered list prefixes, HTML tags, and common HTML entities (`&amp;`, `&lt;`, `&gt;`).
3. **Whitespace collapse** — all whitespace sequences (newlines, tabs, multiple spaces) are collapsed to single spaces, and leading/trailing whitespace is stripped.

This normalization eliminates systematic format-dependent bias. Without it, models that produce markdown output (Mistral OCR, Docling) would incur inflated CER/WER from formatting characters that do not represent OCR errors. The normalization is applied identically to both prediction and ground truth text before any metric computation.
