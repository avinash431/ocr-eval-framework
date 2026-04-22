"""Generate consensus-based ground truth from multiple OCR model outputs.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
Strategy:
- Use Mistral OCR as the primary candidate (highest Phase 1 accuracy)
- Cross-validate against other model outputs using token-level F1
- Compute a consensus confidence score based on agreement
- Output _gt.txt files for documents without existing ground truth

IMPORTANT: Ground truth generated this way should be flagged as
"model-consensus GT" in the whitepaper — it is NOT human-verified.
The circularity caveat applies: Mistral will score perfectly against
its own output used as GT. This is mitigated by cross-validation
and should be disclosed in methodology.
"""

import argparse
from collections import Counter
from pathlib import Path


def compute_token_f1(text_a: str, text_b: str) -> float:
    """Compute token-level F1 between two texts."""
    tokens_a = Counter(text_a.lower().split())
    tokens_b = Counter(text_b.lower().split())

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    shared = set(tokens_a.keys()) & set(tokens_b.keys())
    tp = sum(min(tokens_a[k], tokens_b[k]) for k in shared)
    total_a = sum(tokens_a.values())
    total_b = sum(tokens_b.values())

    precision = tp / total_a if total_a else 0
    recall = tp / total_b if total_b else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_model_outputs(doc_stem: str, results_dir: Path) -> dict[str, str]:
    """Load all model outputs for a given document stem."""
    outputs = {}
    for rd in sorted(results_dir.iterdir()):
        raw_dir = rd / "raw_outputs"
        if not raw_dir.exists():
            continue
        for f in raw_dir.glob(f"*__{doc_stem}.txt"):
            model = f.stem.split("__", 1)[0]
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text and len(text) > 10:  # Skip empty/trivial outputs
                outputs[model] = text
    return outputs


def select_best_candidate(outputs: dict[str, str]) -> tuple[str, str, float]:
    """Select the best GT candidate and compute consensus confidence.

    Priority: mistral_ocr > sarvam_ocr > surya > docling > paddleocr > got_ocr > tesseract

    Returns:
        (selected_model, text, consensus_score)
    """
    priority = ["mistral_ocr", "sarvam_ocr", "surya", "docling",
                "paddleocr", "got_ocr", "tesseract"]

    # Pick the highest-priority model available
    selected_model = None
    for model in priority:
        if model in outputs:
            selected_model = model
            break

    if not selected_model:
        # Fallback to whatever is available
        selected_model = next(iter(outputs))

    candidate_text = outputs[selected_model]

    # Compute consensus: average F1 of candidate against all other models
    other_scores = []
    for model, text in outputs.items():
        if model != selected_model:
            f1 = compute_token_f1(candidate_text, text)
            other_scores.append((model, f1))

    consensus = sum(s for _, s in other_scores) / len(other_scores) if other_scores else 0.0

    return selected_model, candidate_text, consensus


def get_existing_gt_stems(gt_dir: Path) -> set[str]:
    """Get document stems that already have ground truth."""
    stems = set()
    for f in gt_dir.rglob("*_gt.txt"):
        stems.add(f.stem.replace("_gt", ""))
    for f in gt_dir.rglob("*.json"):
        stems.add(f.stem.replace("_gt_structured", "").replace("_gt", ""))
    return stems


def main():
    parser = argparse.ArgumentParser(description="Generate consensus ground truth")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory with model outputs")
    parser.add_argument("--dataset-dir", default="test-dataset",
                        help="Dataset directory")
    parser.add_argument("--gt-dir", default="test-dataset/ground_truth",
                        help="Ground truth output directory")
    parser.add_argument("--min-consensus", type=float, default=0.3,
                        help="Minimum consensus F1 to accept GT (default: 0.3)")
    parser.add_argument("--min-models", type=int, default=3,
                        help="Minimum number of models required (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing")
    parser.add_argument("--categories", nargs="+",
                        default=["02_complex_tables/financial",
                                 "02_complex_tables/multi_column",
                                 "06_mixed_content/equations_formulas"],
                        help="Categories to generate GT for")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)
    gt_dir = Path(args.gt_dir)
    existing_gt = get_existing_gt_stems(gt_dir)

    total_generated = 0
    total_skipped = 0
    total_existing = 0

    for category in args.categories:
        cat_path = dataset_dir / category
        if not cat_path.exists():
            print(f"⚠ Category not found: {category}")
            continue

        # Determine GT subdirectory
        cat_parts = category.split("/")
        gt_subdir = gt_dir / cat_parts[0]
        gt_subdir.mkdir(parents=True, exist_ok=True)

        docs = sorted(cat_path.iterdir())
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(docs)} docs)")
        print(f"{'='*60}")

        for doc in docs:
            if doc.name.startswith("."):
                continue

            stem = doc.stem

            # Skip if GT already exists
            if stem in existing_gt:
                total_existing += 1
                continue

            # Load model outputs
            outputs = load_model_outputs(stem, results_dir)

            if len(outputs) < args.min_models:
                print(f"  SKIP {stem[:50]:50s} — only {len(outputs)} models (need {args.min_models})")
                total_skipped += 1
                continue

            # Select best candidate
            selected_model, text, consensus = select_best_candidate(outputs)

            if consensus < args.min_consensus:
                print(f"  SKIP {stem[:50]:50s} — consensus {consensus:.2f} < {args.min_consensus}")
                total_skipped += 1
                continue

            gt_file = gt_subdir / f"{stem}_gt.txt"

            if args.dry_run:
                print(f"  WOULD {stem[:50]:50s} — {selected_model}, consensus={consensus:.2f}, "
                      f"{len(text)} chars, {len(outputs)} models")
            else:
                gt_file.write_text(text, encoding="utf-8")
                print(f"  ✅ {stem[:50]:50s} — {selected_model}, consensus={consensus:.2f}, "
                      f"{len(text)} chars")

            total_generated += 1

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Generated: {total_generated}")
    print(f"  Skipped (low consensus/models): {total_skipped}")
    print(f"  Already had GT: {total_existing}")
    print(f"{'='*60}")

    if not args.dry_run and total_generated > 0:
        print("\n⚠ IMPORTANT: This GT is model-consensus based (primary: Mistral OCR).")
        print("  Mistral will score CER≈0 against its own output.")
        print("  Cross-model metrics remain valid for ranking other models.")
        print("  Flag as 'model-consensus GT' in whitepaper methodology.")


if __name__ == "__main__":
    main()
