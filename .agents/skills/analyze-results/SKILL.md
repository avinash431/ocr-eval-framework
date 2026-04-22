---
name: analyze-results
description: Analyze and compare OCR evaluation results across models, categories, and document types. Use when the user wants to understand benchmark results, compare model performance, find the best model for a use case, investigate errors, or generate insights from evaluation data. Triggers on "analyze results", "which model is best", "compare performance", "show metrics", "why did X fail", or any question about OCR evaluation outcomes.
---

# Analyze Results

Deep-dive into OCR evaluation results to extract actionable insights.

## Before You Start

1. Locate the results directory — check `results/` for the latest run:
   ```bash
   ls -lt results/ | head -5
   ```
2. Read `batch_summary.json` for the high-level overview
3. Read per-model metrics from `metrics/<model>_metrics.json` for document-level detail

## Analysis Workflows

### Overall Model Comparison

Read `batch_summary.json` and present a ranked table:

```bash
python cli/evaluate.py --results-dir results/<run_dir>
```

Rank models by the user's priority — common orderings:
- **Accuracy-first**: Sort by F1 descending, then CER ascending
- **Speed-first**: Sort by avg latency ascending
- **Cost-first**: Sort by estimated cost ascending
- **Balanced**: Weighted score combining F1, latency, and cost

### Per-Category Analysis

Documents are organized by category (e.g., `01_printed_english/invoices`, `03_handwritten/hindi`). Break down metrics by category to find:

- Which models excel at printed vs. handwritten text
- Which models handle Indian languages best
- Which models struggle with complex tables or low-quality scans

Read individual metric files and group by `get_document_category()` from the document paths.

### Error Investigation

When a model fails or produces poor results:

1. Check the raw output in `raw_outputs/<model>__<doc>.txt`
2. Compare against ground truth in the dataset's `ground_truth/` directory
3. Look for patterns:
   - Empty output → model failed to detect any text
   - Garbled text → wrong language setting or image quality issue
   - Missing tables → model doesn't support structured extraction
   - High CER but decent F1 → formatting/whitespace differences rather than content errors

### Cost-Performance Trade-off

For cloud API models, combine accuracy with cost data:

| Model | F1 | CER | Cost/1000 pages | Value Score |
|-------|-----|-----|-----------------|-------------|
| Use `estimate_cost()` from each model class | | | | F1 / cost |

### Export for External Analysis

```bash
python cli/evaluate.py --results-dir results/<run_dir> --export-csv
```

This creates `all_metrics.csv` which can be loaded into pandas, Excel, or any visualization tool.

## Presenting Findings

When sharing results with the user:

1. Lead with the key finding ("Surya has the best forms CER, but Mistral OCR has 100% success rate across all categories")
2. Show the comparison table with relevant metrics
3. Call out surprising results or outliers
4. Recommend the best model for the user's specific use case
5. Suggest follow-up experiments if results are inconclusive

## Common Questions and How to Answer Them

- **"Which model is best?"** → Depends on document type, language, budget. Show trade-offs.
- **"Why is CER > 1.0?"** → Predicted text is much longer than ground truth (insertions).
- **"Why does model X fail on Y?"** → Check raw output, compare with working model's output on same doc.
- **"Can I trust these numbers?"** → Check sample size, ground truth quality, and metric distribution (mean vs. median).
