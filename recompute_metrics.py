"""Recompute metrics for all models against expanded ground truth.

Reads raw outputs from results/ directories, matches against GT files,
computes all metrics, and outputs per-model and per-category aggregates.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from utils.metrics import compute_all_metrics


def load_gt_text(gt_path: Path) -> str:
    """Load ground truth text from .txt or .json file."""
    if gt_path.suffix == '.json':
        data = json.loads(gt_path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'form' in data:
            words = [w['text'] for w in data['form'] if 'text' in w]
            return ' '.join(words)
        return json.dumps(data)
    return gt_path.read_text(encoding='utf-8')


def get_category(doc_stem: str, dataset_dir: Path) -> str:
    """Determine which category a document belongs to."""
    for cat_dir in dataset_dir.rglob('*'):
        if cat_dir.is_file() and cat_dir.stem == doc_stem:
            # Return relative path from dataset_dir minus the filename
            rel = cat_dir.relative_to(dataset_dir)
            parts = rel.parts[:-1]  # remove filename
            return '/'.join(parts)
    return 'unknown'


def main():
    project_root = Path('.')
    dataset_dir = project_root / 'test-dataset'
    gt_dir = dataset_dir / 'ground_truth'
    results_dir = project_root / 'results'
    output_dir = project_root / 'results' / 'expanded_gt_metrics'
    output_dir.mkdir(exist_ok=True)

    # Build GT map
    gt_map = {}
    for f in gt_dir.rglob('*_gt.txt'):
        stem = f.stem.replace('_gt', '')
        gt_map[stem] = f
    for f in gt_dir.rglob('*.json'):
        if '_structured' in f.name:
            continue
        stem = f.stem
        if stem not in gt_map:
            gt_map[stem] = f

    # Build category map by scanning dataset
    cat_map = {}
    for cat_dir in dataset_dir.iterdir():
        if not cat_dir.is_dir() or cat_dir.name in ('ground_truth', '.DS_Store'):
            continue
        for sub in cat_dir.rglob('*'):
            if sub.is_file() and not sub.name.startswith('.'):
                rel = sub.relative_to(dataset_dir)
                parts = rel.parts[:-1]
                cat_map[sub.stem] = '/'.join(parts)

    # Collect model outputs
    model_outputs = {}
    for rd in sorted(results_dir.iterdir()):
        raw_dir = rd / 'raw_outputs'
        if not raw_dir.exists():
            continue
        for f in raw_dir.glob('*.txt'):
            parts = f.stem.split('__')
            if len(parts) >= 2:
                model = parts[0]
                doc = parts[-1]
                model_outputs.setdefault(model, {})[doc] = f

    # Phase 1 models to evaluate
    models_to_eval = ['tesseract', 'mistral_ocr', 'surya', 'sarvam_ocr', 'docling', 'paddleocr', 'got_ocr']

    all_results = []
    model_summaries = {}

    for model in models_to_eval:
        if model not in model_outputs:
            print(f'  SKIP {model} — no raw outputs found')
            continue

        outputs = model_outputs[model]
        matched_docs = {doc: path for doc, path in outputs.items() if doc in gt_map}

        if not matched_docs:
            print(f'  SKIP {model} — no GT matches')
            continue

        print(f'\n{"="*60}')
        print(f'{model}: {len(matched_docs)} docs with GT')
        print(f'{"="*60}')

        per_doc_results = []
        category_results = defaultdict(list)

        for doc_stem, pred_path in sorted(matched_docs.items()):
            pred_text = pred_path.read_text(encoding='utf-8', errors='replace').strip()
            gt_text = load_gt_text(gt_map[doc_stem])
            category = cat_map.get(doc_stem, 'unknown')

            r = compute_all_metrics(pred_text, gt_text, doc_path=doc_stem, model_name=model)
            per_doc_results.append(r)
            category_results[category].append(r)
            all_results.append({
                'model': model,
                'doc': doc_stem,
                'category': category,
                **r.to_dict()
            })

        # Compute aggregates
        n = len(per_doc_results)
        avg_cer = sum(r.cer for r in per_doc_results) / n
        avg_wer = sum(r.wer for r in per_doc_results) / n
        avg_f1 = sum(r.f1 for r in per_doc_results) / n
        avg_prec = sum(r.precision for r in per_doc_results) / n
        avg_rec = sum(r.recall for r in per_doc_results) / n
        avg_wa = sum(r.word_accuracy for r in per_doc_results) / n
        avg_ed = sum(r.edit_dist for r in per_doc_results) / n
        total_subs = sum(r.char_substitutions for r in per_doc_results)
        total_ins = sum(r.char_insertions for r in per_doc_results)
        total_dels = sum(r.char_deletions for r in per_doc_results)
        total_errors = total_subs + total_ins + total_dels

        model_summaries[model] = {
            'n': n,
            'avg_cer': round(avg_cer, 4),
            'avg_wer': round(avg_wer, 4),
            'avg_f1': round(avg_f1, 4),
            'avg_precision': round(avg_prec, 4),
            'avg_recall': round(avg_rec, 4),
            'avg_word_accuracy': round(avg_wa, 4),
            'avg_edit_dist': round(avg_ed, 4),
            'total_errors': total_errors,
            'substitutions': total_subs,
            'insertions': total_ins,
            'deletions': total_dels,
            'sub_pct': round(total_subs / max(total_errors, 1) * 100, 1),
            'ins_pct': round(total_ins / max(total_errors, 1) * 100, 1),
            'del_pct': round(total_dels / max(total_errors, 1) * 100, 1),
        }

        print(f'  Avg CER: {avg_cer:.4f}')
        print(f'  Avg WER: {avg_wer:.4f}')
        print(f'  Avg F1:  {avg_f1:.4f}  (P={avg_prec:.4f}, R={avg_rec:.4f})')
        print(f'  Avg Word Accuracy: {avg_wa:.4f}')
        print(f'  Avg Edit Distance: {avg_ed:.4f}')
        print(f'  Errors: {total_errors} total (S={total_subs} [{model_summaries[model]["sub_pct"]}%], '
              f'I={total_ins} [{model_summaries[model]["ins_pct"]}%], '
              f'D={total_dels} [{model_summaries[model]["del_pct"]}%])')

        # Per-category breakdown
        print('\n  Category breakdown:')
        for cat in sorted(category_results):
            cr = category_results[cat]
            cn = len(cr)
            cat_cer = sum(r.cer for r in cr) / cn
            cat_f1 = sum(r.f1 for r in cr) / cn
            cat_prec = sum(r.precision for r in cr) / cn
            cat_rec = sum(r.recall for r in cr) / cn
            print(f'    {cat:45s} n={cn:2d}  CER={cat_cer:.4f}  F1={cat_f1:.4f}  P={cat_prec:.4f}  R={cat_rec:.4f}')

    # Write summary JSON
    summary_path = output_dir / 'model_summaries.json'
    with open(summary_path, 'w') as f:
        json.dump(model_summaries, f, indent=2)
    print(f'\n\nSummary written to: {summary_path}')

    # Write per-doc CSV
    csv_path = output_dir / 'per_doc_metrics.csv'
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f'Per-doc CSV written to: {csv_path}')

    # Print comparison table
    print(f'\n{"="*80}')
    print('CROSS-MODEL COMPARISON (expanded GT)')
    print(f'{"="*80}')
    print(f'{"Model":15s} {"n":>4s} {"CER":>8s} {"WER":>8s} {"F1":>8s} {"Prec":>8s} {"Rec":>8s} {"WordAcc":>8s}')
    print('-' * 80)
    for model in models_to_eval:
        if model not in model_summaries:
            continue
        s = model_summaries[model]
        print(f'{model:15s} {s["n"]:4d} {s["avg_cer"]:8.4f} {s["avg_wer"]:8.4f} '
              f'{s["avg_f1"]:8.4f} {s["avg_precision"]:8.4f} {s["avg_recall"]:8.4f} '
              f'{s["avg_word_accuracy"]:8.4f}')

    print(f'\n{"="*80}')
    print('ERROR DECOMPOSITION')
    print(f'{"="*80}')
    print(f'{"Model":15s} {"n":>4s} {"Total":>8s} {"S(%)":>8s} {"I(%)":>8s} {"D(%)":>8s}')
    print('-' * 80)
    for model in models_to_eval:
        if model not in model_summaries:
            continue
        s = model_summaries[model]
        print(f'{model:15s} {s["n"]:4d} {s["total_errors"]:8d} '
              f'{s["sub_pct"]:7.1f}% {s["ins_pct"]:7.1f}% {s["del_pct"]:7.1f}%')


if __name__ == '__main__':
    main()
