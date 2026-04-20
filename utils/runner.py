"""Batch execution engine — runs models across documents with progress tracking."""

import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from models import get_model, list_models
from models.base import OCRResult
from utils.helpers import find_documents, get_document_category, get_ground_truth
from utils.metrics import compute_all_metrics


class EvalRunner:
    """Orchestrates running OCR models on documents and collecting results."""

    def __init__(self, config: dict):
        self.config = config
        self.dataset_dir = config["paths"]["dataset_dir"]
        self.results_base = config["paths"]["results_dir"]
        self.gt_dir = config["paths"].get("ground_truth_dir", "")

    def _make_run_dir(self, tag: str = "") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{tag}" if tag else ts
        run_dir = Path(self.results_base) / name
        (run_dir / "raw_outputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        return run_dir

    def run_single(self, model_name: str, image_path: str) -> OCRResult:
        """Run one model on one document."""
        model = get_model(model_name, self.config)
        print(f"Running {model.display_name} on {Path(image_path).name}...")
        result = model.ocr(image_path)

        if result.success:
            print(f"  ✅ {len(result.raw_text)} chars extracted in {result.latency_ms:.0f}ms")
        else:
            print(f"  ❌ Error: {result.error}")

        # Check for ground truth
        if self.gt_dir:
            gt = get_ground_truth(image_path, self.gt_dir)
            if gt:
                metrics = compute_all_metrics(result.raw_text, gt, image_path, model_name)
                print(f"  📊 CER: {metrics.cer:.4f} | WER: {metrics.wer:.4f} | F1: {metrics.f1:.4f}")

        model.teardown()
        return result

    def run_model(self, model_name: str, doc_paths: list = None,
                  resume_dir: str | None = None) -> Path:
        """Run one model on all documents. Returns path to results.

        If resume_dir is provided, reuses that run directory and skips any
        document whose raw_outputs file already exists — lets a killed run
        pick up exactly where it stopped.
        """
        if resume_dir:
            run_dir = Path(resume_dir)
            (run_dir / "raw_outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        else:
            run_dir = self._make_run_dir(model_name)
        docs = doc_paths or find_documents(self.dataset_dir)

        if not docs:
            print(f"⚠ No documents found in {self.dataset_dir}")
            return run_dir

        model = get_model(model_name, self.config)
        print(f"\n{'='*60}")
        print(f"  Running: {model.display_name}")
        print(f"  Documents: {len(docs)}")
        print(f"  Results: {run_dir}")
        if resume_dir:
            print("  Mode: RESUME (skip docs with existing raw_output)")
        print(f"{'='*60}\n")

        results_path = run_dir / f"{model_name}_results.json"
        metrics_path = run_dir / "metrics" / f"{model_name}_metrics.json"

        if resume_dir and results_path.exists():
            try:
                with open(results_path) as f:
                    results = json.load(f)
            except Exception:
                results = []
        else:
            results = []

        if resume_dir and metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics_list = json.load(f)
            except Exception:
                metrics_list = []
        else:
            metrics_list = []

        already_done = {r.get("document_path") for r in results if r.get("success")}

        for doc_path in tqdm(docs, desc=model.display_name):
            cat_slug = get_document_category(str(doc_path)).replace("/", "_")
            out_file = run_dir / "raw_outputs" / f"{model_name}__{cat_slug}__{doc_path.stem}.txt"
            resolved = str(Path(doc_path).resolve())

            if resume_dir and (resolved in already_done or out_file.exists()):
                continue

            result = model.ocr(str(doc_path))
            result_dict = result.to_dict()
            result_dict["category"] = get_document_category(str(doc_path))
            results.append(result_dict)

            if self.config.get("execution", {}).get("save_raw_output", True) and result.success:
                out_file.write_text(result.raw_text, encoding="utf-8")

            if self.gt_dir:
                gt = get_ground_truth(str(doc_path), self.gt_dir)
                if gt and result.success:
                    m = compute_all_metrics(result.raw_text, gt, str(doc_path), model_name)
                    metrics_list.append(m.to_dict())

            # Flush after every doc so a mid-batch kill doesn't discard progress.
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if metrics_list:
                with open(metrics_path, "w") as f:
                    json.dump(metrics_list, f, indent=2)

        # Print summary
        successful = sum(1 for r in results if r["success"])
        avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        print(f"\n  ✅ {successful}/{len(results)} documents processed")
        print(f"  ⏱  Avg latency: {avg_latency:.0f}ms per document")

        if metrics_list:
            avg_cer = sum(m["cer"] for m in metrics_list) / len(metrics_list)
            avg_wer = sum(m["wer"] for m in metrics_list) / len(metrics_list)
            avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)
            print(f"  📊 Avg CER: {avg_cer:.4f} | WER: {avg_wer:.4f} | F1: {avg_f1:.4f}")
            print(f"     (computed on {len(metrics_list)} docs with ground truth)")

        model.teardown()
        return run_dir

    def run_batch(self, model_names: list = None) -> Path:
        """Run multiple models on all documents."""
        run_dir = self._make_run_dir("batch")
        docs = find_documents(self.dataset_dir)
        models_to_run = model_names or list_models()

        if not docs:
            print(f"⚠ No documents found in {self.dataset_dir}")
            return run_dir

        print(f"\n{'='*60}")
        print("  BATCH EVALUATION")
        print(f"  Models: {len(models_to_run)}")
        print(f"  Documents: {len(docs)}")
        print(f"  Total runs: {len(models_to_run) * len(docs)}")
        print(f"  Results: {run_dir}")
        print(f"{'='*60}\n")

        all_summaries = []

        for model_name in models_to_run:
            print(f"\n--- {model_name} ---")
            try:
                model = get_model(model_name, self.config)
            except Exception as e:
                print(f"  ⚠ Skipping {model_name}: {e}")
                all_summaries.append({
                    "model": model_name, "status": "skipped", "error": str(e)
                })
                continue

            results = []
            metrics_list = []

            for doc_path in tqdm(docs, desc=model_name, leave=False):
                result = model.ocr(str(doc_path))
                rd = result.to_dict()
                rd["category"] = get_document_category(str(doc_path))
                results.append(rd)

                if self.config.get("execution", {}).get("save_raw_output", True) and result.success:
                    cat_slug = get_document_category(str(doc_path)).replace("/", "_")
                    out_file = run_dir / "raw_outputs" / f"{model_name}__{cat_slug}__{doc_path.stem}.txt"
                    out_file.write_text(result.raw_text, encoding="utf-8")

                if self.gt_dir:
                    gt = get_ground_truth(str(doc_path), self.gt_dir)
                    if gt and result.success:
                        m = compute_all_metrics(result.raw_text, gt, str(doc_path), model_name)
                        metrics_list.append(m.to_dict())

            # Save per-model results
            with open(run_dir / f"{model_name}_results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if metrics_list:
                with open(run_dir / "metrics" / f"{model_name}_metrics.json", "w") as f:
                    json.dump(metrics_list, f, indent=2)

            successful = sum(1 for r in results if r["success"])
            avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
            summary = {
                "model": model_name, "status": "completed",
                "total": len(results), "successful": successful,
                "avg_latency_ms": round(avg_latency, 2),
            }
            if metrics_list:
                summary["avg_cer"] = round(sum(m["cer"] for m in metrics_list) / len(metrics_list), 4)
                summary["avg_wer"] = round(sum(m["wer"] for m in metrics_list) / len(metrics_list), 4)
                summary["avg_f1"] = round(sum(m["f1"] for m in metrics_list) / len(metrics_list), 4)
                summary["docs_with_gt"] = len(metrics_list)

            all_summaries.append(summary)
            print(f"  ✅ {successful}/{len(results)} docs | {avg_latency:.0f}ms avg")
            model.teardown()

        # Save batch summary
        with open(run_dir / "batch_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

        # Print final table
        print(f"\n{'='*60}")
        print("  BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Model':<22} {'Success':>8} {'Latency':>10} {'CER':>8} {'WER':>8} {'F1':>8}")
        print(f"  {'-'*70}")
        for s in all_summaries:
            if s["status"] == "skipped":
                print(f"  {s['model']:<22} SKIPPED — {s.get('error', '')[:40]}")
            else:
                cer = f"{s.get('avg_cer', '-')}" if 'avg_cer' in s else "   -"
                wer = f"{s.get('avg_wer', '-')}" if 'avg_wer' in s else "   -"
                f1 = f"{s.get('avg_f1', '-')}" if 'avg_f1' in s else "   -"
                print(f"  {s['model']:<22} {s['successful']:>3}/{s['total']:<4} {s['avg_latency_ms']:>8.0f}ms {cer:>8} {wer:>8} {f1:>8}")

        return run_dir
