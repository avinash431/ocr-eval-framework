"""Tests for utils/metrics.py — OCR evaluation metrics.

Covers: normalize_ocr_text, compute_cer, compute_wer, compute_f1,
compute_edit_distance, compute_error_decomposition, compute_all_metrics.
"""

import pytest
from utils.metrics import (
    normalize_ocr_text,
    compute_cer,
    compute_wer,
    compute_f1,
    compute_edit_distance,
    compute_error_decomposition,
    compute_all_metrics,
    MetricsResult,
)


# ---------------------------------------------------------------------------
# normalize_ocr_text
# ---------------------------------------------------------------------------

class TestNormalizeOcrText:
    def test_empty_string(self):
        assert normalize_ocr_text("") == ""

    def test_none_like_empty(self):
        """Empty/whitespace-only input returns empty."""
        assert normalize_ocr_text("   ") == ""
        assert normalize_ocr_text("\n\t\n") == ""

    def test_whitespace_collapse(self):
        assert normalize_ocr_text("hello   world") == "hello world"
        assert normalize_ocr_text("a\n\nb\tc") == "a b c"

    def test_unicode_nfkc(self):
        # NFKC normalizes full-width chars to ASCII
        assert normalize_ocr_text("\uff21\uff22\uff23") == "ABC"

    def test_strip_markdown_headings(self):
        # Only the # markers are stripped, heading text is preserved
        assert normalize_ocr_text("## Heading\ntext") == "Heading text"
        assert normalize_ocr_text("# H1\n## H2\nfoo") == "H1 H2 foo"

    def test_strip_bold_italic(self):
        assert normalize_ocr_text("**bold**") == "bold"
        assert normalize_ocr_text("*italic*") == "italic"
        assert normalize_ocr_text("***both***") == "both"

    def test_strip_inline_code(self):
        assert normalize_ocr_text("`code`") == "code"

    def test_strip_links(self):
        assert normalize_ocr_text("[click here](https://example.com)") == "click here"

    def test_strip_images(self):
        assert normalize_ocr_text("![alt](image.png)") == ""

    def test_strip_html_tags(self):
        assert normalize_ocr_text("<b>bold</b>") == "bold"
        assert normalize_ocr_text("a <br/> b") == "a b"

    def test_strip_html_entities(self):
        assert normalize_ocr_text("A &amp; B") == "A & B"
        assert normalize_ocr_text("&lt;tag&gt;") == "<tag>"

    def test_strip_table_rows(self):
        assert normalize_ocr_text("| a | b |") == ""

    def test_strip_list_bullets(self):
        assert normalize_ocr_text("- item one\n- item two") == "item one item two"
        assert normalize_ocr_text("* bullet") == "bullet"

    def test_strip_numbered_lists(self):
        assert normalize_ocr_text("1. first\n2. second") == "first second"

    def test_strip_horizontal_rules(self):
        assert normalize_ocr_text("---") == ""
        assert normalize_ocr_text("***") == ""

    def test_preserves_plain_text(self):
        text = "Hello World 123"
        assert normalize_ocr_text(text) == text

    def test_combined_markdown(self):
        md = "# Title\n\n**Bold** and *italic* with `code`\n\n- list item"
        result = normalize_ocr_text(md)
        assert "Bold" in result
        assert "italic" in result
        assert "code" in result
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result or result.count("*") == 0


# ---------------------------------------------------------------------------
# compute_cer
# ---------------------------------------------------------------------------

class TestComputeCer:
    def test_identical(self):
        assert compute_cer("hello", "hello") == 0.0

    def test_completely_different(self):
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0  # 3 substitutions / 3 chars

    def test_empty_both(self):
        assert compute_cer("", "") == 0.0

    def test_empty_gt_nonempty_pred(self):
        assert compute_cer("abc", "") == 1.0

    def test_empty_pred_nonempty_gt(self):
        cer = compute_cer("", "abc")
        assert cer == 1.0  # 3 deletions / 3 chars

    def test_cer_can_exceed_one(self):
        """CER > 1.0 when prediction is much longer than GT."""
        cer = compute_cer("abcdefghij", "ab")
        assert cer > 1.0

    def test_single_char_substitution(self):
        cer = compute_cer("cat", "bat")
        assert abs(cer - 1/3) < 0.01

    def test_insertion(self):
        cer = compute_cer("abcd", "abc")
        assert abs(cer - 1/3) < 0.01


# ---------------------------------------------------------------------------
# compute_wer
# ---------------------------------------------------------------------------

class TestComputeWer:
    def test_identical(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        wer = compute_wer("foo bar", "baz qux")
        assert wer == 1.0

    def test_empty_both(self):
        assert compute_wer("", "") == 0.0

    def test_empty_gt_nonempty_pred(self):
        assert compute_wer("some words", "") == 1.0

    def test_empty_pred_nonempty_gt(self):
        assert compute_wer("", "some words") == 1.0

    def test_one_word_wrong(self):
        wer = compute_wer("the cat sat", "the dog sat")
        assert abs(wer - 1/3) < 0.01

    def test_extra_words(self):
        """WER > 1.0 when prediction has many more words."""
        wer = compute_wer("a b c d e f", "a")
        assert wer > 1.0


# ---------------------------------------------------------------------------
# compute_f1 (Counter-based)
# ---------------------------------------------------------------------------

class TestComputeF1:
    def test_identical(self):
        f1, p, r = compute_f1("hello world", "hello world")
        assert f1 == 1.0
        assert p == 1.0
        assert r == 1.0

    def test_no_overlap(self):
        f1, p, r = compute_f1("abc", "xyz")
        assert f1 == 0.0
        assert p == 0.0
        assert r == 0.0

    def test_empty_both(self):
        f1, _p, _r = compute_f1("", "")
        assert f1 == 1.0

    def test_empty_gt_nonempty_pred(self):
        f1, _p, _r = compute_f1("hello", "")
        assert f1 == 0.0

    def test_empty_pred_nonempty_gt(self):
        f1, _p, _r = compute_f1("", "hello")
        assert f1 == 0.0

    def test_counter_not_set(self):
        """Counter-based F1 must penalize missing duplicates.

        With set-based F1, "hello hello hello world" vs "hello world"
        would give F1=1.0 (both sets are {hello, world}).
        Counter-based should give F1 < 1.0.
        """
        f1, p, r = compute_f1("hello hello hello world", "hello world")
        assert f1 < 1.0
        # pred has 4 tokens, gt has 2. TP=2 (min counts). P=2/4=0.5, R=2/2=1.0
        assert abs(p - 0.5) < 0.01
        assert abs(r - 1.0) < 0.01
        # F1 = 2*0.5*1.0/(0.5+1.0) = 2/3
        assert abs(f1 - 2/3) < 0.01

    def test_partial_overlap(self):
        _f1, p, r = compute_f1("the quick brown", "the slow brown fox")
        # pred: {the:1, quick:1, brown:1}, gt: {the:1, slow:1, brown:1, fox:1}
        # TP = min(the) + min(brown) = 2. P=2/3, R=2/4=0.5
        assert abs(p - 2/3) < 0.01
        assert abs(r - 0.5) < 0.01

    def test_case_insensitive(self):
        f1, _, _ = compute_f1("Hello World", "hello world")
        assert f1 == 1.0

    def test_precision_recall_distinction(self):
        """Over-generation: high recall, low precision."""
        _f1, p, r = compute_f1("a b c d e", "a b")
        assert r == 1.0  # all GT tokens found
        assert p < 1.0   # extra tokens in prediction
        assert abs(p - 2/5) < 0.01

    def test_under_extraction(self):
        """Under-extraction: high precision, low recall."""
        _f1, p, r = compute_f1("a", "a b c d")
        assert p == 1.0  # all predicted tokens correct
        assert abs(r - 0.25) < 0.01


# ---------------------------------------------------------------------------
# compute_edit_distance
# ---------------------------------------------------------------------------

class TestComputeEditDistance:
    def test_identical(self):
        assert compute_edit_distance("hello", "hello") == 0.0

    def test_empty_both(self):
        assert compute_edit_distance("", "") == 0.0

    def test_completely_different(self):
        dist = compute_edit_distance("abc", "xyz")
        assert dist == 1.0  # 3 subs / max(3, 3) = 1.0

    def test_normalized_zero_to_one(self):
        dist = compute_edit_distance("hello", "hallo")
        assert 0.0 < dist < 1.0

    def test_different_lengths(self):
        dist = compute_edit_distance("ab", "abcdef")
        # 4 insertions / max(2, 6) = 4/6
        assert abs(dist - 4/6) < 0.01


# ---------------------------------------------------------------------------
# compute_error_decomposition
# ---------------------------------------------------------------------------

class TestComputeErrorDecomposition:
    def test_identical(self):
        s, i, d = compute_error_decomposition("hello", "hello")
        assert s == 0
        assert i == 0
        assert d == 0

    def test_empty_both(self):
        s, i, d = compute_error_decomposition("", "")
        assert (s, i, d) == (0, 0, 0)

    def test_pure_substitution(self):
        s, i, d = compute_error_decomposition("cat", "bat")
        assert s == 1
        assert i == 0
        assert d == 0

    def test_pure_insertion(self):
        """Extra char in prediction = insertion error from OCR perspective."""
        s, i, d = compute_error_decomposition("abcd", "abc")
        assert i == 1  # 'd' in pred but not in GT
        assert s == 0
        assert d == 0

    def test_pure_deletion(self):
        """Missing char from prediction = deletion error from OCR perspective."""
        s, i, d = compute_error_decomposition("abc", "abcd")
        assert d == 1  # 'd' in GT but not in pred
        assert s == 0
        assert i == 0

    def test_mixed_errors(self):
        s, i, d = compute_error_decomposition("axyz", "abcd")
        total = s + i + d
        assert total > 0

    def test_total_equals_levenshtein(self):
        """Total S+I+D should equal the Levenshtein distance."""
        from rapidfuzz.distance import Levenshtein
        pred, gt = "kitten", "sitting"
        s, i, d = compute_error_decomposition(pred, gt)
        dist = Levenshtein.distance(pred, gt)
        assert s + i + d == dist


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_metrics_result(self):
        result = compute_all_metrics("hello world", "hello world")
        assert isinstance(result, MetricsResult)

    def test_identical_text(self):
        r = compute_all_metrics("hello world", "hello world")
        assert r.cer == 0.0
        assert r.wer == 0.0
        assert r.f1 == 1.0
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.edit_dist == 0.0
        assert r.word_accuracy == 1.0
        assert r.char_substitutions == 0
        assert r.char_insertions == 0
        assert r.char_deletions == 0

    def test_normalization_applied(self):
        """Markdown should be stripped before metrics."""
        r = compute_all_metrics("**hello** world", "hello world")
        assert r.cer == 0.0
        assert r.f1 == 1.0

    def test_doc_path_and_model_name(self):
        r = compute_all_metrics("a", "b", doc_path="/test.png", model_name="test_model")
        assert r.doc_path == "/test.png"
        assert r.model_name == "test_model"

    def test_char_counts(self):
        r = compute_all_metrics("abc", "abcde")
        assert r.char_count_pred == 3
        assert r.char_count_gt == 5

    def test_word_counts(self):
        r = compute_all_metrics("one two", "one two three")
        assert r.word_count_pred == 2
        assert r.word_count_gt == 3

    def test_word_accuracy_floored_at_zero(self):
        """word_accuracy = max(1 - WER, 0), should never go negative."""
        r = compute_all_metrics("a b c d e f g", "x")
        assert r.word_accuracy == 0.0

    def test_error_rates_normalized(self):
        r = compute_all_metrics("cat", "bat")
        # After normalization both are 3 chars, 1 substitution
        assert r.substitution_rate is not None
        assert r.substitution_rate > 0
        total_rate = r.substitution_rate + r.insertion_rate + r.deletion_rate
        assert abs(total_rate - r.cer) < 0.01

    def test_to_dict(self):
        r = compute_all_metrics("hello", "hello")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "cer" in d
        assert "f1" in d
        assert d["cer"] == 0.0

    def test_to_dict_rounds_floats(self):
        r = compute_all_metrics("abc", "abcdef")
        d = r.to_dict()
        # All float values should be rounded to 4 decimal places
        for k, v in d.items():
            if isinstance(v, float):
                assert v == round(v, 4), f"{k}={v} not rounded to 4dp"

    def test_empty_prediction_and_gt(self):
        r = compute_all_metrics("", "")
        assert r.cer == 0.0
        assert r.f1 == 1.0
        assert r.char_count_pred == 0
        assert r.char_count_gt == 0

    @pytest.mark.parametrize("pred,gt", [
        ("hello", "hello"),
        ("abc", "xyz"),
        ("", "test"),
        ("test", ""),
        ("hello hello hello", "hello"),
    ])
    def test_metrics_are_finite(self, pred, gt):
        """All metric values should be finite numbers."""
        r = compute_all_metrics(pred, gt)
        for k, v in r.to_dict().items():
            if isinstance(v, (int, float)):
                assert v == v, f"{k} is NaN"  # NaN != NaN
