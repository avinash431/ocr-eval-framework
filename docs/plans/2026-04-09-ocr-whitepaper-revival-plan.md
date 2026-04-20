# OCR Whitepaper Revival Plan

> For Hermes: execute this plan incrementally from the repo root. Prefer producing concrete artifacts in `results/`, `docs/`, and `analysis/` instead of keeping progress in chat.

Goal: finish the OCR whitepaper with evidence that matches the repository’s actual evaluation coverage, and avoid over-claiming unsupported results.

Architecture: use the existing `ocr-eval-framework` as the source of truth for experiments, the Excel tracker as the coordination log, and a new markdown draft as the writing artifact. First reconcile scope vs reality, then produce enough benchmark evidence to support a credible internal whitepaper, then draft and polish.

Tech stack: Python 3.12 venv, existing OCR wrappers in `models/`, evaluation scripts (`run_single.py`, `run_model.py`, `run_batch.py`, `evaluate.py`), tracker workbook in `docs/`, markdown draft files in `docs/whitepaper/`.

---

## Current state snapshot

Verified on 2026-04-09:
- Repo loads all 14 model wrappers successfully via `python run_batch.py --list`.
- Local Tesseract baseline runs successfully on `test-dataset/02_complex_tables/forms/0012199830.png`.
- Current committed dataset has 101 documents, not the broader 100-150 cross-language corpus claimed in the outline.
- Current dataset coverage is narrow:
  - `02_complex_tables`: 45 docs
  - `03_handwritten`: 30 docs
  - `04_indian_languages`: 1 doc
  - `06_mixed_content`: 25 docs
- Tracker workbook contains only two logged runs:
  - Tesseract on `0012199830.png`
  - Mistral OCR on `0012199830.png`
- No batch results directory currently exists from earlier full-model runs.
- The framework can support the paper, but the paper cannot honestly claim broad conclusions yet.

## Non-negotiable principle

The final whitepaper must align with actual evidence. You have two valid paths:

1. Evidence-first path (recommended): run enough experiments to support a meaningful comparison.
2. Scope-first path: reduce the paper’s claims to reflect the currently evaluated subset.

Do not keep the current mismatch where the narrative promises broad multilingual enterprise coverage but the repo evidence does not yet support it.

---

## Phase 1: Reconcile scope with reality

### Task 1: Create a factual inventory artifact
Objective: capture the real current repo state in a reusable markdown file.

Files:
- Create: `docs/whitepaper/current-state.md`

Step 1: Summarize the current dataset and tracker status.
Include:
- dataset counts by category
- models registered
- models actually run
- known runnable baseline command
- open gaps in language/category coverage

Step 2: Save the summary in markdown.
Suggested headings:
- Framework status
- Dataset coverage
- Existing results
- Risks to paper credibility
- Recommended next action

Verification:
- File exists at `docs/whitepaper/current-state.md`
- It contains the numbers verified above

### Task 2: Decide paper scope
Objective: choose what paper you are actually finishing.

Files:
- Modify: `docs/whitepaper/current-state.md`
- Create later: `docs/whitepaper/draft.md`

Choose one of these scopes:

Option A — Full benchmark paper
- Compare as many of the 14 models as are practically runnable
- Expand/verify missing dataset coverage enough to support multilingual and structured-extraction claims
- Stronger paper, more work

Option B — Interim internal whitepaper
- Frame the paper as “Phase 1 benchmark framework and early findings”
- Compare a smaller subset of models with strong methodological detail
- Faster and still credible internally

Recommendation: start with Option B unless your team can actually provide model/API access this week.

Verification:
- The chosen scope is written explicitly at the top of `docs/whitepaper/current-state.md`

---

## Phase 2: Produce defensible evaluation evidence

### Task 3: Lock a minimum viable comparison set
Objective: pick the first models that can realistically be run soon.

Files:
- Create: `docs/whitepaper/model-run-plan.md`

Recommended minimum set:
- `tesseract` — CPU baseline, already verified
- `mistral_ocr` — strongest known partial result from tracker
- `paddleocr` — lightweight open-source comparator
- `docling` — structure-aware document conversion
- `surya` — multilingual open-source OCR

Nice-to-have additions:
- `sarvam_ocr` — India-focused document intelligence
- `qwen_vl` or `olmocr` if hardware permits

Avoid blocking on all models before writing anything.

Verification:
- `docs/whitepaper/model-run-plan.md` lists model, owner, credential/hardware requirement, and expected execution status

### Task 4: Generate baseline single-run sanity checks
Objective: verify each chosen model can run at least one common sample before full-dataset evaluation.

Files:
- Create under `results/` automatically
- Update workbook: `docs/OCR_Test_Dataset_Tracker.xlsx`

Commands:
- `source venv/bin/activate && python run_single.py --model tesseract --input test-dataset/02_complex_tables/forms/0012199830.png`
- `source venv/bin/activate && python run_single.py --model mistral_ocr --input test-dataset/02_complex_tables/forms/0012199830.png`
- Repeat for each selected model

Record for each run:
- success/failure
- latency
- CER/WER/F1 if ground truth exists
- qualitative notes

Verification:
- Every selected model has either a successful sanity-run log or a documented blocker

### Task 5: Run full-dataset evaluations for the minimum set
Objective: produce real comparison artifacts.

Files:
- Create under `results/<timestamp>_<model>/`
- Metrics JSON under `results/<timestamp>_<model>/metrics/`

Commands:
- `source venv/bin/activate && python run_model.py --model tesseract`
- `source venv/bin/activate && python run_model.py --model mistral_ocr`
- `source venv/bin/activate && python run_model.py --model paddleocr`
- `source venv/bin/activate && python run_model.py --model docling`
- Optionally one cloud API comparator

Verification:
- Each successful run creates:
  - `<model>_results.json`
  - `metrics/<model>_metrics.json`
  - raw outputs in `raw_outputs/`

### Task 6: Generate machine-readable comparison exports
Objective: turn per-model outputs into reportable comparison artifacts.

Files:
- Create: `analysis/` directory if needed
- Create: `results/<run_dir>/report.html`
- Create optionally: `results/<run_dir>/all_metrics.csv`

Preferred approach:
- If enough models are rerun together, use `run_batch.py` and then:
  - `source venv/bin/activate && python evaluate.py --results-dir results/<batch_run> --export-csv`
- If runs are separate, write a small aggregation script later in `analysis/` to combine metrics across model run dirs

Verification:
- You can open one HTML report and one CSV with cross-document metrics

---

## Phase 3: Close the biggest credibility gaps

### Task 7: Fix narrative claims that exceed evidence
Objective: remove unsupported claims from the paper outline.

Files:
- Create: `docs/whitepaper/draft.md`
- Reference: `docs/OCR_Whitepaper_Plan_v2.docx`

Mandatory adjustments if dataset remains as-is:
- Do not claim broad European language evaluation
- Do not claim robust internal-document coverage if internal docs are absent from the committed dataset
- Do not claim complete 14-model comparison unless most models are actually evaluated
- Reframe “10 dimensions” carefully if some dimensions are still qualitative/not measured by code

Verification:
- Every claim in the draft can be traced to either a results artifact or an explicitly qualitative note

### Task 8: Tighten methodology section
Objective: make the methodology defensible even if scope is narrower than originally planned.

Files:
- Modify/Create: `docs/whitepaper/draft.md`

Methodology must clearly state:
- exact dataset size used in final analysis
- category distribution
- which models were fully run vs partially tested vs planned-only
- which metrics are automated vs manually assessed
- environment limitations (API access, GPU, licensing)

Verification:
- A reviewer can understand what was actually tested without opening the code

### Task 9: Build the comparative evidence tables
Objective: create the tables that the whitepaper needs.

Files:
- Create: `docs/whitepaper/tables.md`

Minimum tables:
1. Model comparison table
   - model
   - type
   - deployment mode
   - status tested / partially tested / not tested
2. Overall metrics table
   - model
   - avg CER
   - avg WER
   - avg F1
   - avg latency
3. Category-wise table
   - model
   - category
   - docs
   - avg CER
   - avg F1
4. Failure analysis table
   - model
   - category/doc
   - failure pattern
   - likely cause

Verification:
- Tables are populated from actual results, not placeholder text

---

## Phase 4: Draft the paper quickly

### Task 10: Write the markdown whitepaper draft
Objective: get a complete first draft into version control.

Files:
- Create: `docs/whitepaper/draft.md`

Recommended section order:
1. Executive summary
2. Problem statement and business need
3. Evaluation methodology
4. Dataset and metrics
5. Model set and test status
6. Results and analysis
7. Failure analysis and limitations
8. Recommendations by use case
9. Next steps / Phase 2 roadmap

Important:
- Write the limitations section before final polish
- Use the single verified Tesseract vs Mistral sample as an illustrative anecdote, not the main conclusion

Verification:
- Draft is complete end-to-end, even if rough

### Task 11: Add visuals
Objective: make the paper presentation-ready.

Files:
- Create: `docs/whitepaper/figures/`
- Create: charts as `.png` or `.svg`

Suggested visuals:
- bar chart of avg F1 by model
- bar chart of avg latency by model
- heatmap by model × category
- side-by-side sample OCR output snippets for 2-3 representative documents

Verification:
- At least 3 visuals are referenced from `draft.md`

### Task 12: Produce the final submission format
Objective: convert the draft into the format your company expects.

Files:
- Could be: `docs/whitepaper/final.docx` or `.pdf`

Approach:
- Either continue in markdown and export later
- Or port the markdown draft into the company whitepaper template once content is stable

Verification:
- Final version exists in the required format and is reviewable without repo context

---

## Immediate execution order for the next working session

If you want the fastest credible path, do these in order:

1. Create `docs/whitepaper/current-state.md`
2. Choose scope: interim paper vs full benchmark
3. Run sanity checks for 3-5 models
4. Run full evaluation for the 3-5 models that actually work
5. Generate comparison artifacts
6. Draft `docs/whitepaper/draft.md`
7. Add limitations and recommendations
8. Polish into submission format

---

## My recommendation for you specifically

Because you left this midway, the highest-leverage path is:

- treat this as an internal “Phase 1 OCR benchmark and framework” whitepaper
- compare 4-5 models well instead of 14 models vaguely
- explicitly position the repo/framework as a reusable benchmarking asset
- include a roadmap section for the remaining models and missing language/document categories

That gets you a credible submission faster and avoids a reviewer catching evidence gaps.

---

## Suggested commit sequence

1. `docs: add OCR whitepaper revival plan`
2. `docs: add current state assessment`
3. `feat: add initial benchmark runs for selected OCR models`
4. `docs: add tables and first whitepaper draft`
5. `docs: polish final OCR whitepaper`
