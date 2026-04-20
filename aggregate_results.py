#!/usr/bin/env python3
"""Aggregate results from multiple model run directories and update whitepaper tables.

Usage:
    python aggregate_results.py

This script scans all results/ directories for completed model runs,
computes aggregate metrics, and prints updated tables for the whitepaper.
It also generates updated charts in docs/whitepaper/figures/.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def find_all_results() -> dict[str, dict]:
    """Find the latest results for each model across all result directories."""
    results_base = Path("results")
    model_results = {}

    for run_dir in sorted(results_base.iterdir()):
        if not run_dir.is_dir():
            continue
        for results_file in run_dir.glob("*_results.json"):
            model_name = results_file.stem.replace("_results", "")
            metrics_file = run_dir / "metrics" / f"{model_name}_metrics.json"

            with open(results_file) as f:
                results = json.load(f)

            metrics = []
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

            # Keep the latest run for each model
            model_results[model_name] = {
                "run_dir": str(run_dir),
                "results": results,
                "metrics": metrics,
            }

    return model_results


def compute_model_summary(model_name: str, data: dict) -> dict:
    """Compute summary statistics for a model."""
    results = data["results"]
    metrics = data["metrics"]

    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    latencies = [r["latency_ms"] for r in results if r.get("success") and r.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    summary = {
        "model": model_name,
        "total": total,
        "successful": successful,
        "success_rate": f"{successful}/{total} ({successful / total * 100:.1f}%)" if total > 0 else "N/A",
        "avg_latency_ms": round(avg_latency),
        "docs_with_gt": len(metrics),
    }

    if metrics:
        summary["avg_cer"] = round(sum(m["cer"] for m in metrics) / len(metrics), 4)
        summary["avg_wer"] = round(sum(m["wer"] for m in metrics) / len(metrics), 4)
        summary["avg_f1"] = round(sum(m["f1"] for m in metrics) / len(metrics), 4)
        summary["avg_bleu"] = round(sum(m["bleu"] for m in metrics) / len(metrics), 4)

    # Category breakdown from results
    cats = defaultdict(lambda: {"total": 0, "success": 0, "latencies": []})
    for r in results:
        cat = r.get("category", "unknown")
        cats[cat]["total"] += 1
        if r.get("success"):
            cats[cat]["success"] += 1
            cats[cat]["latencies"].append(r["latency_ms"])
    summary["categories"] = dict(cats)

    # Category breakdown from metrics
    met_cats = defaultdict(list)
    for m in metrics:
        p = m["doc_path"]
        if "forms" in p:
            cat = "forms"
        elif "receipts" in p:
            cat = "receipts"
        else:
            cat = "other"
        met_cats[cat].append(m)
    summary["metric_categories"] = dict(met_cats)

    return summary


def print_table3(summaries: list[dict]) -> None:
    """Print Table 3: Full-Dataset Results."""
    print("\n## Table 3: Full-Dataset Results\n")
    print("| Model | Success Rate | Avg Latency (ms) | Avg CER | Avg WER | Avg F1 | Avg BLEU | Docs with GT |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        cer = f"{s.get('avg_cer', '-')}" if "avg_cer" in s else "—"
        wer = f"{s.get('avg_wer', '-')}" if "avg_wer" in s else "—"
        f1 = f"{s.get('avg_f1', '-')}" if "avg_f1" in s else "—"
        bleu = f"{s.get('avg_bleu', '-')}" if "avg_bleu" in s else "—"
        print(
            f"| {s['model']} | {s['success_rate']} | {s['avg_latency_ms']:,} | "
            f"{cer} | {wer} | {f1} | {bleu} | {s['docs_with_gt']} |"
        )


def print_category_tables(summaries: list[dict]) -> None:
    """Print category-wise metric tables."""
    for cat_name in ["forms", "receipts"]:
        print(f"\n## Category: {cat_name}\n")
        print("| Model | Avg CER | Avg WER | Avg F1 | n |")
        print("|:---|---:|---:|---:|---:|")
        for s in summaries:
            docs = s.get("metric_categories", {}).get(cat_name, [])
            if docs:
                n = len(docs)
                avg_cer = sum(d["cer"] for d in docs) / n
                avg_wer = sum(d["wer"] for d in docs) / n
                avg_f1 = sum(d["f1"] for d in docs) / n
                print(f"| {s['model']} | {avg_cer:.4f} | {avg_wer:.4f} | {avg_f1:.4f} | {n} |")
            else:
                print(f"| {s['model']} | — | — | — | 0 |")


def print_success_rates(summaries: list[dict]) -> None:
    """Print category-level success rates."""
    print("\n## Category-Level Success Rates\n")
    all_cats = set()
    for s in summaries:
        all_cats.update(s.get("categories", {}).keys())

    cats_sorted = sorted(all_cats)
    header = "| Category |" + " | ".join(s["model"] for s in summaries) + " |"
    separator = "|:---|" + " | ".join("---:" for _ in summaries) + " |"
    print(header)
    print(separator)

    for cat in cats_sorted:
        row = f"| {cat} |"
        for s in summaries:
            c = s.get("categories", {}).get(cat)
            if c:
                rate = f"{c['success']}/{c['total']}"
            else:
                row += " — |"
                continue
            row += f" {rate} |"
        print(row)


def generate_charts(summaries: list[dict]) -> None:
    """Generate updated comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np  # noqa: F401 — used by plt internals
    except ImportError:
        print("\n[WARN] matplotlib not available, skipping chart generation")
        return

    fig_dir = Path("docs/whitepaper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    # Full-dataset F1 comparison
    models_with_f1 = [s for s in summaries if "avg_f1" in s]
    if len(models_with_f1) >= 2:
        names = [s["model"] for s in models_with_f1]
        f1_vals = [s["avg_f1"] for s in models_with_f1]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, f1_vals, color=colors[: len(names)], edgecolor="white")
        ax.set_xlabel("Average Token F1", fontsize=12)
        ax.set_title(
            "Full-Dataset Average F1 by Model (Ground-Truth Subset)", fontsize=14, fontweight="bold"
        )
        for bar, val in zip(bars, f1_vals):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10
            )
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig7_full_dataset_f1.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\n[INFO] Generated fig7_full_dataset_f1.png")

    # Success rate comparison
    names = [s["model"] for s in summaries]
    success_rates = [s["successful"] / s["total"] * 100 if s["total"] > 0 else 0 for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, success_rates, color=colors[: len(names)], edgecolor="white")
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("Full-Dataset Success Rate by Model", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    for bar, val in zip(bars, success_rates):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_full_dataset_success_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[INFO] Generated fig8_full_dataset_success_rate.png")


def main() -> None:
    print("=" * 60)
    print("  OCR Evaluation Results Aggregator")
    print("=" * 60)

    model_results = find_all_results()

    if not model_results:
        print("\nNo results found in results/ directory.")
        sys.exit(1)

    print(f"\nFound results for {len(model_results)} models:")
    for name, data in sorted(model_results.items()):
        print(f"  {name}: {len(data['results'])} docs, {len(data['metrics'])} with GT ({data['run_dir']})")

    # Phase 1 models only
    phase1_models = ["tesseract", "mistral_ocr", "docling", "paddleocr", "surya", "sarvam_ocr"]
    summaries = []
    for model in phase1_models:
        if model in model_results:
            s = compute_model_summary(model, model_results[model])
            summaries.append(s)

    print_table3(summaries)
    print_category_tables(summaries)
    print_success_rates(summaries)
    generate_charts(summaries)

    print("\n" + "=" * 60)
    print("  Done. Copy the tables above into docs/whitepaper/tables.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
