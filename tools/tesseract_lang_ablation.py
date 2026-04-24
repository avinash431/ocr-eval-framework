#!/usr/bin/env python3
"""Tesseract language-pack ablation on the 15 USGov forms subset.

Runs Tesseract twice per document — once with an English-only language pack
(``eng``) and once with the multi-language pack currently configured in
``configs/config.yaml`` (``eng+hin+tel+tam+ben``). Emits per-document and
aggregate CER/WER/F1/latency deltas so the whitepaper can quantify the
"multi-language config garbles Indic characters" claim with real numbers.

Outputs a JSON report and a markdown summary under
``results/<timestamp>_tesseract_lang_ablation/``.

Usage:
    source venv/bin/activate
    python tools/tesseract_lang_ablation.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.helpers import load_config  # noqa: E402
from utils.metrics import compute_all_metrics, normalize_ocr_text  # noqa: E402
from models.tesseract_model import TesseractOCR  # noqa: E402


FORMS_DIR = REPO_ROOT / "test-dataset" / "02_complex_tables" / "forms"
GT_DIR = REPO_ROOT / "test-dataset" / "ground_truth" / "02_complex_tables"

LANG_VARIANTS = {
    "eng_only": "eng",
    "multi_lang": "eng+hin+tel+tam+ben",
}


@dataclass
class PerDocResult:
    doc_stem: str
    lang_variant: str
    lang_string: str
    success: bool
    latency_ms: float
    char_count: int
    cer: Optional[float]
    wer: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    error: Optional[str] = None


def discover_forms_with_gt() -> list[tuple[Path, Path]]:
    """Find the subset of forms that have `_gt.txt` files available.

    Returns a list of (form_path, gt_path) tuples. The 5 original PNG forms
    use JSON-structured GT rather than `_gt.txt`, so this ablation is scoped
    to the 15 USGov PDFs (which have plain-text ground truth suitable for
    CER/WER).
    """
    pairs: list[tuple[Path, Path]] = []
    for form_path in sorted(FORMS_DIR.iterdir()):
        if form_path.is_dir() or form_path.name.startswith('.'):
            continue
        gt_path = GT_DIR / f"{form_path.stem}_gt.txt"
        if gt_path.exists():
            pairs.append((form_path, gt_path))
    return pairs


def run_variant(model: TesseractOCR, doc_path: Path, gt_text: str,
                lang_variant: str, lang_string: str) -> PerDocResult:
    """Execute Tesseract on a single doc with a specific language pack."""
    model.config["tesseract"]["lang"] = lang_string
    start = time.perf_counter()
    try:
        ocr_result = model._ocr_impl(str(doc_path))
        latency_ms = (time.perf_counter() - start) * 1000.0
        pred = ocr_result.raw_text or ""
        metrics = compute_all_metrics(pred, gt_text)
        return PerDocResult(
            doc_stem=doc_path.stem,
            lang_variant=lang_variant,
            lang_string=lang_string,
            success=True,
            latency_ms=latency_ms,
            char_count=len(normalize_ocr_text(pred)),
            cer=metrics.cer,
            wer=metrics.wer,
            f1=metrics.f1,
            precision=metrics.precision,
            recall=metrics.recall,
        )
    except Exception as exc:  # noqa: BLE001 — want every runtime failure logged
        latency_ms = (time.perf_counter() - start) * 1000.0
        return PerDocResult(
            doc_stem=doc_path.stem,
            lang_variant=lang_variant,
            lang_string=lang_string,
            success=False,
            latency_ms=latency_ms,
            char_count=0,
            cer=None, wer=None, f1=None, precision=None, recall=None,
            error=str(exc),
        )


def aggregate(variant_rows: list[PerDocResult]) -> dict:
    ok = [r for r in variant_rows if r.success]
    def _mean(key: str) -> Optional[float]:
        vals = [getattr(r, key) for r in ok if getattr(r, key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None
    return {
        "n": len(variant_rows),
        "successful": len(ok),
        "mean_cer": _mean("cer"),
        "mean_wer": _mean("wer"),
        "mean_f1": _mean("f1"),
        "mean_precision": _mean("precision"),
        "mean_recall": _mean("recall"),
        "mean_latency_ms": round(statistics.fmean([r.latency_ms for r in ok]), 1) if ok else None,
    }


def paired_deltas(per_doc: list[PerDocResult]) -> dict:
    """Compute per-doc deltas (multi_lang - eng_only) across shared successes."""
    by_doc: dict[str, dict[str, PerDocResult]] = {}
    for row in per_doc:
        by_doc.setdefault(row.doc_stem, {})[row.lang_variant] = row
    pairs = [(d["eng_only"], d["multi_lang"]) for d in by_doc.values()
             if "eng_only" in d and "multi_lang" in d
             and d["eng_only"].success and d["multi_lang"].success]
    if not pairs:
        return {"n_paired": 0}
    deltas = {
        "cer": [m.cer - e.cer for e, m in pairs if e.cer is not None and m.cer is not None],
        "wer": [m.wer - e.wer for e, m in pairs if e.wer is not None and m.wer is not None],
        "f1":  [m.f1  - e.f1  for e, m in pairs if e.f1  is not None and m.f1  is not None],
        "latency_ms": [m.latency_ms - e.latency_ms for e, m in pairs],
    }
    return {
        "n_paired": len(pairs),
        "mean_delta_cer_multi_minus_eng": round(statistics.fmean(deltas["cer"]), 4) if deltas["cer"] else None,
        "mean_delta_wer_multi_minus_eng": round(statistics.fmean(deltas["wer"]), 4) if deltas["wer"] else None,
        "mean_delta_f1_multi_minus_eng":  round(statistics.fmean(deltas["f1"]),  4) if deltas["f1"]  else None,
        "mean_delta_latency_ms_multi_minus_eng": round(statistics.fmean(deltas["latency_ms"]), 1),
    }


def write_markdown_summary(out_path: Path, agg: dict, deltas: dict,
                           per_doc: list[PerDocResult]) -> None:
    lines = [
        "# Tesseract Language-Pack Ablation",
        "",
        f"Dataset: {agg['eng_only']['n']} USGov PDF forms with `_gt.txt` ground truth.",
        "",
        "## Aggregate Metrics",
        "",
        "| Variant | n | Successful | Mean CER | Mean WER | Mean F1 | Mean Latency (ms) |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("eng_only", "multi_lang"):
        v = agg[variant]
        lines.append(
            f"| {variant} (`{LANG_VARIANTS[variant]}`) | {v['n']} | {v['successful']} | "
            f"{v['mean_cer']} | {v['mean_wer']} | {v['mean_f1']} | {v['mean_latency_ms']} |"
        )
    lines += [
        "",
        "## Paired Deltas (multi_lang − eng_only)",
        "",
        f"- Paired docs: {deltas.get('n_paired', 0)}",
        f"- Δ CER: {deltas.get('mean_delta_cer_multi_minus_eng')}",
        f"- Δ WER: {deltas.get('mean_delta_wer_multi_minus_eng')}",
        f"- Δ F1: {deltas.get('mean_delta_f1_multi_minus_eng')}",
        f"- Δ Latency (ms): {deltas.get('mean_delta_latency_ms_multi_minus_eng')}",
        "",
        "## Per-Document Results",
        "",
        "| Doc stem | Variant | Success | CER | WER | F1 | Latency (ms) |",
        "|:---|:---|:---:|---:|---:|---:|---:|",
    ]
    for row in per_doc:
        lines.append(
            f"| {row.doc_stem} | {row.lang_variant} | {'✓' if row.success else '✗'} | "
            f"{row.cer} | {row.wer} | {row.f1} | {row.latency_ms:.1f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config_path = REPO_ROOT / "configs" / "config.local.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))
    config.setdefault("tesseract", {})

    pairs = discover_forms_with_gt()
    if not pairs:
        print("ERROR: no forms with _gt.txt found under ground_truth/02_complex_tables/",
              file=sys.stderr)
        return 1
    print(f"Ablation scope: {len(pairs)} USGov PDF forms with human-verified GT")

    model = TesseractOCR(config)
    model.setup()

    per_doc: list[PerDocResult] = []
    for variant_key, lang_string in LANG_VARIANTS.items():
        print(f"\n=== variant: {variant_key} (lang='{lang_string}') ===")
        for doc_path, gt_path in pairs:
            gt_text = gt_path.read_text(encoding="utf-8")
            row = run_variant(model, doc_path, gt_text, variant_key, lang_string)
            per_doc.append(row)
            status = "ok" if row.success else f"FAIL({row.error!r})"
            print(f"  {doc_path.name:40s} CER={row.cer} F1={row.f1} "
                  f"lat={row.latency_ms:.0f}ms [{status}]")

    agg = {
        "eng_only":   aggregate([r for r in per_doc if r.lang_variant == "eng_only"]),
        "multi_lang": aggregate([r for r in per_doc if r.lang_variant == "multi_lang"]),
    }
    deltas = paired_deltas(per_doc)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / f"{ts}_tesseract_lang_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": ts,
        "scope": "02_complex_tables/forms — USGov PDFs with _gt.txt",
        "variants": LANG_VARIANTS,
        "aggregate": agg,
        "paired_deltas_multi_minus_eng": deltas,
        "per_doc": [asdict(r) for r in per_doc],
    }
    (out_dir / "lang_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown_summary(out_dir / "lang_ablation.md", agg, deltas, per_doc)

    print(f"\nWrote: {out_dir}/lang_ablation.json")
    print(f"Wrote: {out_dir}/lang_ablation.md")
    print("\nAggregate:")
    print(json.dumps(agg, indent=2))
    print("Paired deltas (multi − eng):")
    print(json.dumps(deltas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
