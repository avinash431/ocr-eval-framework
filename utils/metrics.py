"""Evaluation metrics for OCR output comparison."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsResult:
    """Computed metrics for one document + model pair."""
    doc_path: str
    model_name: str
    cer: Optional[float] = None       # Character Error Rate (lower is better)
    wer: Optional[float] = None       # Word Error Rate (lower is better)
    bleu: Optional[float] = None      # BLEU score (higher is better)
    edit_dist: Optional[float] = None  # Normalized edit distance (lower is better)
    f1: Optional[float] = None        # Token-level F1 (higher is better)
    char_count_pred: int = 0
    char_count_gt: int = 0
    word_count_pred: int = 0
    word_count_gt: int = 0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def compute_cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    from rapidfuzz.distance import Levenshtein
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    dist = Levenshtein.distance(prediction, ground_truth)
    return dist / max(len(ground_truth), 1)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate."""
    from rapidfuzz.distance import Levenshtein
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    dist = Levenshtein.distance(pred_words, gt_words)
    return dist / max(len(gt_words), 1)


def compute_bleu(prediction: str, ground_truth: str) -> float:
    """BLEU score (sentence level)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = ground_truth.split()
        hyp = prediction.split()
        if not ref or not hyp:
            return 0.0
        smoothie = SmoothingFunction().method1
        return sentence_bleu([ref], hyp, smoothing_function=smoothie)
    except Exception:
        return 0.0


def compute_edit_distance(prediction: str, ground_truth: str) -> float:
    """Normalized edit distance (0 = identical, 1 = completely different)."""
    from rapidfuzz.distance import Levenshtein
    if not ground_truth and not prediction:
        return 0.0
    dist = Levenshtein.distance(prediction, ground_truth)
    return dist / max(len(prediction), len(ground_truth), 1)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(prediction: str, ground_truth: str,
                        doc_path: str = "", model_name: str = "") -> MetricsResult:
    """Compute all metrics for a prediction/ground truth pair."""
    # Normalize whitespace
    pred = " ".join(prediction.split())
    gt = " ".join(ground_truth.split())

    return MetricsResult(
        doc_path=doc_path,
        model_name=model_name,
        cer=compute_cer(pred, gt),
        wer=compute_wer(pred, gt),
        bleu=compute_bleu(pred, gt),
        edit_dist=compute_edit_distance(pred, gt),
        f1=compute_f1(pred, gt),
        char_count_pred=len(pred),
        char_count_gt=len(gt),
        word_count_pred=len(pred.split()),
        word_count_gt=len(gt.split()),
    )
