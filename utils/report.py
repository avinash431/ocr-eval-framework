"""Generate HTML evaluation report from results."""

import json
from pathlib import Path
from datetime import datetime


def generate_report(results_dir: str) -> str:
    """Generate an HTML comparison report from batch results."""
    results_dir = Path(results_dir)
    summary_file = results_dir / "batch_summary.json"

    if not summary_file.exists():
        print("⚠ No batch_summary.json found. Run a batch evaluation first.")
        return ""

    with open(summary_file) as f:
        summaries = json.load(f)

    # Load per-model metrics
    model_metrics = {}
    metrics_dir = results_dir / "metrics"
    if metrics_dir.exists():
        for mf in metrics_dir.glob("*_metrics.json"):
            model_name = mf.stem.replace("_metrics", "")
            with open(mf) as f:
                model_metrics[model_name] = json.load(f)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>OCR Evaluation Report</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    h1 {{ color: #1B3A5C; border-bottom: 3px solid #2E75B6; padding-bottom: 10px; }}
    h2 {{ color: #2E75B6; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th {{ background: #1B3A5C; color: white; padding: 12px; text-align: left; }}
    td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
    tr:hover {{ background: #f0f7ff; }}
    .good {{ color: #2e7d32; font-weight: bold; }}
    .medium {{ color: #f57f17; font-weight: bold; }}
    .bad {{ color: #c62828; font-weight: bold; }}
    .skipped {{ color: #999; font-style: italic; }}
    .metric-bar {{ display: inline-block; height: 16px; border-radius: 3px; }}
    .summary-box {{ display: inline-block; background: #E8F0F8; padding: 15px 25px; border-radius: 6px; margin: 5px; text-align: center; }}
    .summary-number {{ font-size: 28px; font-weight: bold; color: #1B3A5C; }}
    .summary-label {{ font-size: 12px; color: #666; }}
    .timestamp {{ color: #999; font-size: 12px; }}
</style>
</head><body>
<div class="container">
    <h1>OCR Model Evaluation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    # Summary boxes
    completed = [s for s in summaries if s["status"] == "completed"]
    total_models = len(summaries)
    total_docs = completed[0]["total"] if completed else 0
    best_f1_model = max(completed, key=lambda s: s.get("avg_f1", 0)) if completed else None
    fastest_model = min(completed, key=lambda s: s.get("avg_latency_ms", 9999)) if completed else None

    html += '<div style="text-align:center; margin: 30px 0;">'
    html += f'<div class="summary-box"><div class="summary-number">{total_models}</div><div class="summary-label">Models Tested</div></div>'
    html += f'<div class="summary-box"><div class="summary-number">{total_docs}</div><div class="summary-label">Documents</div></div>'
    if best_f1_model:
        html += f'<div class="summary-box"><div class="summary-number">{best_f1_model["model"]}</div><div class="summary-label">Best F1: {best_f1_model.get("avg_f1", "N/A")}</div></div>'
    if fastest_model:
        html += f'<div class="summary-box"><div class="summary-number">{fastest_model["model"]}</div><div class="summary-label">Fastest: {fastest_model.get("avg_latency_ms", 0):.0f}ms</div></div>'
    html += '</div>'

    # Main comparison table
    html += '<h2>Overall Comparison</h2>'
    html += '<table><tr><th>Model</th><th>Success Rate</th><th>Avg Latency</th><th>CER ↓</th><th>WER ↓</th><th>F1 ↑</th><th>BLEU ↑</th></tr>'

    for s in sorted(summaries, key=lambda x: x.get("avg_f1", 0), reverse=True):
        if s["status"] == "skipped":
            html += f'<tr class="skipped"><td>{s["model"]}</td><td colspan="6">Skipped: {s.get("error", "")[:60]}</td></tr>'
            continue

        success_pct = s["successful"] / s["total"] * 100 if s["total"] else 0
        latency = s.get("avg_latency_ms", 0)
        cer = s.get("avg_cer")
        wer = s.get("avg_wer")
        f1 = s.get("avg_f1")

        cer_class = "good" if cer and cer < 0.05 else "medium" if cer and cer < 0.15 else "bad" if cer else ""
        f1_class = "good" if f1 and f1 > 0.85 else "medium" if f1 and f1 > 0.70 else "bad" if f1 else ""

        html += f'<tr>'
        html += f'<td><strong>{s["model"]}</strong></td>'
        html += f'<td>{success_pct:.0f}% ({s["successful"]}/{s["total"]})</td>'
        html += f'<td>{latency:.0f}ms</td>'
        html += f'<td class="{cer_class}">{cer:.4f}</td>' if cer is not None else '<td>-</td>'
        html += f'<td>{wer:.4f}</td>' if wer is not None else '<td>-</td>'
        html += f'<td class="{f1_class}">{f1:.4f}</td>' if f1 is not None else '<td>-</td>'
        html += f'<td>-</td>'
        html += f'</tr>'

    html += '</table>'

    # Per-category breakdown if metrics available
    if model_metrics:
        html += '<h2>Performance by Category</h2>'
        categories = set()
        for mm in model_metrics.values():
            for m in mm:
                cat = "/".join(Path(m["doc_path"]).parts[-3:-1])
                categories.add(cat)

        for cat in sorted(categories):
            html += f'<h3>{cat}</h3><table>'
            html += '<tr><th>Model</th><th>CER ↓</th><th>WER ↓</th><th>F1 ↑</th><th>Docs</th></tr>'
            for model_name, mm in model_metrics.items():
                cat_metrics = [m for m in mm if cat in m["doc_path"]]
                if cat_metrics:
                    avg_cer = sum(m["cer"] for m in cat_metrics) / len(cat_metrics)
                    avg_wer = sum(m["wer"] for m in cat_metrics) / len(cat_metrics)
                    avg_f1 = sum(m["f1"] for m in cat_metrics) / len(cat_metrics)
                    html += f'<tr><td>{model_name}</td><td>{avg_cer:.4f}</td><td>{avg_wer:.4f}</td><td>{avg_f1:.4f}</td><td>{len(cat_metrics)}</td></tr>'
            html += '</table>'

    html += '</div></body></html>'

    # Save report
    report_path = results_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"📊 Report saved: {report_path}")
    return str(report_path)
