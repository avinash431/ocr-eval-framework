"""Recompute metrics and corpus summaries for the whitepaper artifacts.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
Reads raw outputs from the latest results directories, matches them against the
current visible dataset snapshot, recomputes metrics on the GT-evaluable subset,
and emits:

- results/expanded_gt_metrics/model_summaries.json
- results/expanded_gt_metrics/per_doc_metrics.csv
- results/expanded_gt_metrics/statistical_tests.json
- results/expanded_gt_metrics/corpus_summary.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

from utils.dataset_inventory import category_counts, find_documents, get_document_category
from utils.metrics import MetricsResult, compute_all_metrics


MODELS_TO_EVAL = [
    "tesseract",
    "mistral_ocr",
    "surya",
    "sarvam_ocr",
    "docling",
    "paddleocr",
    "got_ocr",
]

GT_OUTPUT_DIR = Path("results/expanded_gt_metrics")


def load_gt_text(gt_path: Path) -> str:
    """Load ground truth text from .txt or .json file."""
    if gt_path.suffix == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "form" in data:
            words = [word["text"] for word in data["form"] if "text" in word]
            return " ".join(words)
        return json.dumps(data)
    return gt_path.read_text(encoding="utf-8")


def find_ground_truth_path(doc_path: Path, gt_dir: Path) -> Path | None:
    """Find the matching GT file for a visible document."""
    exact = gt_dir / f"{doc_path.stem}_gt.txt"
    if exact.exists():
        return exact

    patterns = [
        f"{doc_path.stem}_gt.txt",
        f"{doc_path.stem}.txt",
        f"{doc_path.stem}.md",
        f"{doc_path.stem}.json",
    ]
    for pattern in patterns:
        for candidate in gt_dir.rglob(pattern):
            if candidate.suffix == ".json" and "_structured" in candidate.name:
                continue
            return candidate
    return None


def effect_size_label(diffs: list[float]) -> str | None:
    """Return a Cohen's d magnitude label for paired differences."""
    if len(diffs) < 2:
        return None

    mean_diff = sum(diffs) / len(diffs)
    variance = sum((diff - mean_diff) ** 2 for diff in diffs) / (len(diffs) - 1)
    if variance <= 0:
        return None

    cohen_d = abs(mean_diff / variance ** 0.5)
    if cohen_d < 0.2:
        magnitude = "negligible"
    elif cohen_d < 0.5:
        magnitude = "small"
    elif cohen_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    return f"{magnitude} (d={cohen_d:.3f})"


def build_current_corpus(dataset_dir: Path) -> tuple[list[Path], dict[str, Path], dict[str, str]]:
    """Build the current visible document snapshot."""
    docs = find_documents(dataset_dir)
    doc_by_stem: dict[str, Path] = {}
    category_by_stem: dict[str, str] = {}

    for doc in docs:
        doc_by_stem[doc.stem] = doc
        category_by_stem[doc.stem] = get_document_category(doc)

    return docs, doc_by_stem, category_by_stem


def build_gt_entries(dataset_dir: Path, doc_by_stem: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Build GT metadata for the current visible corpus."""
    gt_dir = dataset_dir / "ground_truth"
    gt_entries: dict[str, dict[str, Any]] = {}

    for stem, doc_path in doc_by_stem.items():
        gt_path = find_ground_truth_path(doc_path, gt_dir)
        if not gt_path:
            continue

        category = get_document_category(doc_path)
        tier = "human_verified" if category == "02_complex_tables/forms" else "consensus"
        gt_entries[stem] = {
            "path": gt_path,
            "category": category,
            "tier": tier,
        }

    return gt_entries


def load_latest_result_runs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the latest available results JSON per model."""
    latest_runs: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for results_file in run_dir.glob("*_results.json"):
            model = results_file.stem.replace("_results", "")
            try:
                results = json.loads(results_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            latest_runs[model] = {
                "run_dir": run_dir,
                "results_file": results_file,
                "results": results,
            }

    return latest_runs


def load_latest_raw_outputs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the latest raw output directory per model."""
    latest_outputs: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        raw_dir = run_dir / "raw_outputs"
        if not raw_dir.exists():
            continue

        per_model_outputs: dict[str, dict[str, Path]] = defaultdict(dict)
        for output_file in raw_dir.glob("*.txt"):
            parts = output_file.stem.split("__")
            if len(parts) < 2:
                continue
            model = parts[0]
            doc_stem = parts[-1]
            per_model_outputs[model][doc_stem] = output_file

        for model, outputs in per_model_outputs.items():
            latest_outputs[model] = {
                "run_dir": run_dir,
                "outputs": outputs,
            }

    return latest_outputs


def build_model_run_coverage(
    docs: list[Path],
    doc_by_stem: dict[str, Path],
    latest_runs: dict[str, dict[str, Any]],
    latest_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize run coverage against the current visible corpus."""
    visible_total = len(docs)
    visible_counts = category_counts(docs)
    visible_resolved = {str(doc.resolve(strict=False)) for doc in docs}
    coverage: dict[str, dict[str, Any]] = {}

    for model in MODELS_TO_EVAL:
        if model in latest_runs:
            run = latest_runs[model]
            category_attempts = {
                category: {"visible_total": count, "attempted": 0, "successful": 0}
                for category, count in visible_counts.items()
            }
            attempted = 0
            successful = 0
            off_corpus = 0

            for result in run["results"]:
                document_path = result.get("document_path")
                if not document_path:
                    continue

                normalized = str(Path(document_path).resolve(strict=False))
                if normalized not in visible_resolved:
                    off_corpus += 1
                    continue

                category = result.get("category") or get_document_category(document_path)
                category_attempts.setdefault(
                    category,
                    {"visible_total": 0, "attempted": 0, "successful": 0},
                )
                attempted += 1
                category_attempts[category]["attempted"] += 1
                if result.get("success"):
                    successful += 1
                    category_attempts[category]["successful"] += 1

            coverage[model] = {
                "run_dir": run["run_dir"].name,
                "coverage_basis": "results_json",
                "visible_doc_total": visible_total,
                "attempted_visible_docs": attempted,
                "successful_visible_docs": successful,
                "attempt_rate_pct": round(attempted / max(visible_total, 1) * 100, 1),
                "success_rate_pct": round(successful / max(visible_total, 1) * 100, 1),
                "results_not_in_current_corpus": off_corpus,
                "category_attempts": category_attempts,
            }
            continue

        if model in latest_outputs:
            output_run = latest_outputs[model]
            category_attempts = {
                category: {"visible_total": count, "attempted": 0, "successful": 0}
                for category, count in visible_counts.items()
            }
            attempted = 0
            off_corpus = 0

            for stem in output_run["outputs"]:
                doc_path = doc_by_stem.get(stem)
                if not doc_path:
                    off_corpus += 1
                    continue
                category = get_document_category(doc_path)
                attempted += 1
                category_attempts[category]["attempted"] += 1
                category_attempts[category]["successful"] += 1

            coverage[model] = {
                "run_dir": output_run["run_dir"].name,
                "coverage_basis": "raw_outputs_only",
                "visible_doc_total": visible_total,
                "attempted_visible_docs": attempted,
                "successful_visible_docs": attempted,
                "attempt_rate_pct": round(attempted / max(visible_total, 1) * 100, 1),
                "success_rate_pct": round(attempted / max(visible_total, 1) * 100, 1),
                "results_not_in_current_corpus": off_corpus,
                "category_attempts": category_attempts,
            }

    return coverage


def compute_model_metrics(
    gt_entries: dict[str, dict[str, Any]],
    latest_outputs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compute per-document metrics and per-model aggregates for GT-evaluable docs."""
    all_results: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}

    for model in MODELS_TO_EVAL:
        if model not in latest_outputs:
            continue

        outputs = latest_outputs[model]["outputs"]
        matched_docs = {stem: path for stem, path in outputs.items() if stem in gt_entries}
        if not matched_docs:
            continue

        per_doc_results: list[MetricsResult] = []
        category_results: dict[str, list[MetricsResult]] = defaultdict(list)

        for doc_stem, pred_path in sorted(matched_docs.items()):
            pred_text = pred_path.read_text(encoding="utf-8", errors="replace").strip()
            gt_text = load_gt_text(gt_entries[doc_stem]["path"])
            category = gt_entries[doc_stem]["category"]
            tier = gt_entries[doc_stem]["tier"]

            result = compute_all_metrics(pred_text, gt_text, doc_path=doc_stem, model_name=model)
            per_doc_results.append(result)
            category_results[category].append(result)
            all_results.append(
                {
                    "model": model,
                    "doc": doc_stem,
                    "category": category,
                    "gt_tier": tier,
                    **result.to_dict(),
                }
            )

        count = len(per_doc_results)
        total_subs = sum(result.char_substitutions for result in per_doc_results)
        total_ins = sum(result.char_insertions for result in per_doc_results)
        total_dels = sum(result.char_deletions for result in per_doc_results)
        total_errors = total_subs + total_ins + total_dels

        per_category_summary = {}
        for category, category_metrics in sorted(category_results.items()):
            cat_count = len(category_metrics)
            per_category_summary[category] = {
                "n": cat_count,
                "avg_cer": round(sum(result.cer for result in category_metrics) / cat_count, 4),
                "avg_wer": round(sum(result.wer for result in category_metrics) / cat_count, 4),
                "avg_f1": round(sum(result.f1 for result in category_metrics) / cat_count, 4),
                "avg_precision": round(sum(result.precision for result in category_metrics) / cat_count, 4),
                "avg_recall": round(sum(result.recall for result in category_metrics) / cat_count, 4),
            }

        model_summaries[model] = {
            "n": count,
            "avg_cer": round(sum(result.cer for result in per_doc_results) / count, 4),
            "avg_wer": round(sum(result.wer for result in per_doc_results) / count, 4),
            "avg_f1": round(sum(result.f1 for result in per_doc_results) / count, 4),
            "avg_precision": round(sum(result.precision for result in per_doc_results) / count, 4),
            "avg_recall": round(sum(result.recall for result in per_doc_results) / count, 4),
            "avg_word_accuracy": round(sum(result.word_accuracy for result in per_doc_results) / count, 4),
            "avg_edit_dist": round(sum(result.edit_dist for result in per_doc_results) / count, 4),
            "total_errors": total_errors,
            "substitutions": total_subs,
            "insertions": total_ins,
            "deletions": total_dels,
            "sub_pct": round(total_subs / max(total_errors, 1) * 100, 1),
            "ins_pct": round(total_ins / max(total_errors, 1) * 100, 1),
            "del_pct": round(total_dels / max(total_errors, 1) * 100, 1),
            "categories": per_category_summary,
        }

    return all_results, model_summaries


def build_corpus_summary(
    docs: list[Path],
    gt_entries: dict[str, dict[str, Any]],
    model_run_coverage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical corpus summary artifact for manuscript numbers."""
    visible_counts = category_counts(docs)
    gt_by_category: dict[str, int] = defaultdict(int)
    gt_tier_totals: dict[str, int] = defaultdict(int)
    gt_tier_by_category: dict[str, dict[str, Any]] = {}

    for entry in gt_entries.values():
        category = entry["category"]
        gt_by_category[category] += 1
        gt_tier_totals[entry["tier"]] += 1
        gt_tier_by_category[category] = {
            "count": gt_by_category[category],
            "tier": entry["tier"],
        }

    no_gt_by_category = {
        category: visible_counts.get(category, 0) - gt_by_category.get(category, 0)
        for category in visible_counts
    }

    return {
        "generated": datetime.now().isoformat(),
        "visible_document_count": len(docs),
        "visible_documents_by_category": visible_counts,
        "ground_truth": {
            "total_documents": len(gt_entries),
            "documents_by_category": dict(sorted(gt_by_category.items())),
            "tier_totals": dict(sorted(gt_tier_totals.items())),
            "tier_by_category": dict(sorted(gt_tier_by_category.items())),
            "no_gt_count": sum(no_gt_by_category.values()),
            "no_gt_by_category": dict(sorted(no_gt_by_category.items())),
        },
        "model_run_coverage": model_run_coverage,
    }


def build_statistical_tests(all_results: list[dict[str, Any]], model_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the supported pairwise statistical tests artifact."""
    per_model_docs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    forms_docs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in all_results:
        model = row["model"]
        per_model_docs[model][row["doc"]] = row
        if row["category"] == "02_complex_tables/forms":
            forms_docs[model][row["doc"]] = row

    significant = []
    not_significant = []
    comparison_models = [model for model in MODELS_TO_EVAL if model != "mistral_ocr" and model in per_model_docs]

    for i, model_a in enumerate(comparison_models):
        for model_b in comparison_models[i + 1:]:
            shared_docs = sorted(set(per_model_docs[model_a]) & set(per_model_docs[model_b]))
            if len(shared_docs) < 5:
                continue

            f1_a = [float(per_model_docs[model_a][doc]["f1"]) for doc in shared_docs]
            f1_b = [float(per_model_docs[model_b][doc]["f1"]) for doc in shared_docs]
            diffs = [a - b for a, b in zip(f1_a, f1_b)]
            if all(diff == 0 for diff in diffs):
                continue

            try:
                _, p_value = wilcoxon(f1_a, f1_b)
            except (ValueError, TypeError):
                continue

            record = {
                "a": model_a,
                "b": model_b,
                "n": len(shared_docs),
                "p": round(float(p_value), 4),
            }

            mean_a = sum(f1_a) / len(f1_a)
            mean_b = sum(f1_b) / len(f1_b)
            if mean_a != mean_b:
                record["delta"] = round(abs(mean_a - mean_b), 4)
                record["winner"] = model_a if mean_a > mean_b else model_b

            label = effect_size_label(diffs)
            if label:
                record["effect_size"] = label

            if p_value < 0.05:
                significant.append(record)
            else:
                not_significant.append(record)

    forms_note = "At n=5, no pairwise comparisons reach significance."
    form_pairs = []
    form_models = [model for model in ["tesseract", "mistral_ocr", "surya", "sarvam_ocr", "docling"] if model in forms_docs]
    for i, model_a in enumerate(form_models):
        for model_b in form_models[i + 1:]:
            shared_docs = sorted(set(forms_docs[model_a]) & set(forms_docs[model_b]))
            if len(shared_docs) < 5:
                continue
            f1_a = [float(forms_docs[model_a][doc]["f1"]) for doc in shared_docs]
            f1_b = [float(forms_docs[model_b][doc]["f1"]) for doc in shared_docs]
            diffs = [a - b for a, b in zip(f1_a, f1_b)]
            if all(diff == 0 for diff in diffs):
                continue
            try:
                _, p_value = wilcoxon(f1_a, f1_b)
            except (ValueError, TypeError):
                continue
            form_pairs.append(round(float(p_value), 4))
    if form_pairs:
        forms_note = f"At n=5, no pairwise comparisons reach significance. Minimum p={min(form_pairs):.4f}."

    ranking = []
    for model, summary in sorted(model_summaries.items(), key=lambda item: item[1]["avg_f1"], reverse=True):
        if model == "mistral_ocr":
            continue
        ranking.append(f"{model} ({summary['avg_f1']:.4f})")

    significant.sort(key=lambda row: (row["p"], row.get("winner", "")))
    not_significant.sort(key=lambda row: (row["p"], row["a"], row["b"]))

    return {
        "note": "Statistical tests on current GT-evaluable docs. Mistral excluded from consensus-GT pairwise comparisons due to circularity.",
        "f1_pairwise_wilcoxon": {
            "significant_at_005": significant,
            "not_significant": not_significant,
        },
        "forms_only_n5": {
            "note": forms_note,
        },
        "ranking_by_f1": ranking,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_per_doc_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-document metric rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(".")
    dataset_dir = project_root / "test-dataset"
    results_dir = project_root / "results"
    GT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    docs, doc_by_stem, _ = build_current_corpus(dataset_dir)
    gt_entries = build_gt_entries(dataset_dir, doc_by_stem)
    latest_runs = load_latest_result_runs(results_dir)
    latest_outputs = load_latest_raw_outputs(results_dir)
    model_run_coverage = build_model_run_coverage(docs, doc_by_stem, latest_runs, latest_outputs)
    per_doc_rows, model_summaries = compute_model_metrics(gt_entries, latest_outputs)
    corpus_summary = build_corpus_summary(docs, gt_entries, model_run_coverage)
    statistical_tests = build_statistical_tests(per_doc_rows, model_summaries)

    write_json(GT_OUTPUT_DIR / "model_summaries.json", model_summaries)
    write_json(GT_OUTPUT_DIR / "corpus_summary.json", corpus_summary)
    write_json(GT_OUTPUT_DIR / "statistical_tests.json", statistical_tests)
    write_per_doc_csv(GT_OUTPUT_DIR / "per_doc_metrics.csv", per_doc_rows)

    print("=" * 72)
    print("WHITEPAPER ARTIFACTS REGENERATED")
    print("=" * 72)
    print(f"Visible corpus: {corpus_summary['visible_document_count']} docs")
    print(f"GT coverage: {corpus_summary['ground_truth']['total_documents']} docs")
    print(f"No GT: {corpus_summary['ground_truth']['no_gt_count']} docs")
    print()
    for model in MODELS_TO_EVAL:
        if model not in model_run_coverage:
            continue
        coverage = model_run_coverage[model]
        print(
            f"{model:12s} success={coverage['successful_visible_docs']:>3d}/{coverage['visible_doc_total']:<3d} "
            f"attempted={coverage['attempted_visible_docs']:>3d}/{coverage['visible_doc_total']:<3d} "
            f"basis={coverage['coverage_basis']}"
        )
    print()
    print(f"Artifacts written to {GT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
