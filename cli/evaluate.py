#!/usr/bin/env python3
"""Compute metrics from results and generate HTML report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from pathlib import Path
from utils.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate results and generate report")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--export-csv", action="store_true", help="Also export metrics as CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Generate HTML report
    report_path = generate_report(str(results_dir))

    # Export CSV if requested
    if args.export_csv:
        metrics_dir = results_dir / "metrics"
        if metrics_dir.exists():
            import pandas as pd
            all_rows = []
            for mf in metrics_dir.glob("*_metrics.json"):
                with open(mf) as f:
                    data = json.load(f)
                all_rows.extend(data)
            if all_rows:
                df = pd.DataFrame(all_rows)
                csv_path = results_dir / "all_metrics.csv"
                df.to_csv(csv_path, index=False)
                print(f"📊 CSV exported: {csv_path}")

    # Print summary table from batch_summary.json
    summary_file = results_dir / "batch_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summaries = json.load(f)

        print(f"\n{'='*80}")
        print(f"  {'Model':<24} {'Status':<12} {'Success':>8} {'Latency':>10} {'CER':>8} {'WER':>8} {'F1':>8}")
        print(f"  {'-'*76}")
        for s in sorted(summaries, key=lambda x: x.get("avg_f1", 0), reverse=True):
            if s["status"] == "skipped":
                print(f"  {s['model']:<24} {'SKIPPED':<12}")
                continue
            cer_s = f"{s['avg_cer']:.4f}" if "avg_cer" in s else "   -"
            wer_s = f"{s['avg_wer']:.4f}" if "avg_wer" in s else "   -"
            f1_s = f"{s['avg_f1']:.4f}" if "avg_f1" in s else "   -"
            print(f"  {s['model']:<24} {'OK':<12} {s['successful']:>3}/{s['total']:<4} {s['avg_latency_ms']:>8.0f}ms {cer_s:>8} {wer_s:>8} {f1_s:>8}")
        print(f"{'='*80}")

    if report_path:
        print(f"\n🌐 Open report in browser: file://{Path(report_path).resolve()}")


if __name__ == "__main__":
    main()
