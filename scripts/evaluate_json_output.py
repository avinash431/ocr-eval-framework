#!/usr/bin/env python3
"""Evaluate OCR prediction JSON against ground truth text or JSON."""

import argparse
import json
from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH if this script is run from scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import compute_all_metrics


def load_prediction(pred_path: Path) -> str:
    text = ""
    if pred_path.suffix.lower() == ".json":
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "blocks" in data and isinstance(data["blocks"], list):
                blocks = sorted(
                    data["blocks"],
                    key=lambda b: b.get("reading_order", 0)
                )
                text = " ".join(str(b.get("text", "")).strip() for b in blocks).strip()
            elif "text" in data and isinstance(data["text"], str):
                text = data["text"].strip()
            elif "pages" in data and isinstance(data["pages"], list):
                text = " ".join(
                    str(page.get("text", "")).strip()
                    for page in data["pages"]
                ).strip()
            else:
                # Fallback: dump all string values in JSON
                text = " ".join(
                    str(v).strip()
                    for v in _extract_strings_from_json(data)
                    if str(v).strip()
                ).strip()
        else:
            text = str(data)
    else:
        text = pred_path.read_text(encoding="utf-8").strip()
    return text


def _extract_strings_from_json(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _extract_strings_from_json(v)
    elif isinstance(value, list):
        for item in value:
            yield from _extract_strings_from_json(item)


def load_ground_truth(gt_path: Path) -> str:
    if gt_path.suffix.lower() == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "form" in data and isinstance(data["form"], list):
            words = []
            for item in data["form"]:
                if isinstance(item, dict):
                    if "words" in item and isinstance(item["words"], list):
                        words.extend(str(w.get("text", "")).strip() for w in item["words"] if isinstance(w, dict))
                    elif "text" in item:
                        words.append(str(item["text"]).strip())
            return " ".join(w for w in words if w).strip()
        if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
            return data["text"].strip()
        return " ".join(str(v).strip() for v in _extract_strings_from_json(data) if str(v).strip()).strip()
    return gt_path.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prediction JSON against a ground truth file.")
    parser.add_argument("--prediction", required=True, help="Path to prediction JSON or text file")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON or text file")
    args = parser.parse_args()

    pred_path = Path(args.prediction)
    gt_path = Path(args.ground_truth)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    prediction = load_prediction(pred_path)
    ground_truth = load_ground_truth(gt_path)

    metrics = compute_all_metrics(prediction, ground_truth,
                                  doc_path=str(gt_path),
                                  model_name="sarvam_ocr")

    print("Prediction text length:", len(prediction))
    print("Ground truth text length:", len(ground_truth))
    print()
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
