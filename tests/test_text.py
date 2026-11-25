"""Text-generation metrics: BLEU, chrF, ROUGE across sacrebleu, nltk, torchmetrics, evaluate, rouge-score."""

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("nltk")
pytest.importorskip("rouge_score")
pytest.importorskip("torchmetrics")

from metric_diff import compare  # noqa: E402
from metric_diff.testing import assert_metrics_agree  # noqa: E402

PREDS = ["The cat sat on the mat.",
         "It is a guide to action, which ensures that the military always obeys the commands of the party.",
         "Hello there!"]
REFS = [["The cat is sitting on the mat."],
        ["It is a guide to action that ensures that the military will forever heed Party commands."],
        ["Hello, there."]]
# At least 4 words each, no punctuation: tokenisation and nltk's short-text rule don't matter.
LONG_P = ["the cat sat on the mat today", "a quick brown fox jumps over the lazy dog",
          "we went to the market in the morning"]
LONG_R = ["the cat sat on the mat", "the quick brown fox jumped over a lazy dog",
          "we walked to the market this morning"]


def keys(r):
    return sorted(c.key for c in r.causes)


def test_bleu_defaults_explained():
    r = compare("bleu", REFS, PREDS)
    assert r.status == "mismatch" and r.confirmed
    assert keys(r) == ["bleu_tokenize", "nltk_short_hypotheses", "text_scale"]
    assert r.results["sacrebleu"].value == pytest.approx(39.16, abs=0.01)
    assert r.results["torchmetrics"].value == pytest.approx(0.3289, abs=1e-4)
    assert "rouge_score" not in r.results  # libraries without the metric are not listed


def test_bleu_same_tokenizer_leaves_only_the_scale():
    r = compare("bleu", REFS, PREDS, tokenize="13a", backends=["sacrebleu", "torchmetrics", "evaluate"])
    assert r.confirmed and keys(r) == ["text_scale"]


def test_bleu_agrees_on_long_whitespace_text():
    r = compare("bleu", LONG_R, LONG_P, backends=["torchmetrics", "nltk", "evaluate"], tokenize="whitespace")
    assert r.agree, str(r)


def test_sentence_bleu_smoothing():
    r = compare("bleu", LONG_R + ["hello there"], LONG_P + ["hi there"], aggregation="sentence",
                tokenize="whitespace", backends=["sacrebleu", "torchmetrics"])
    assert r.confirmed and keys(r) == ["bleu_smoothing", "text_scale"]


def test_smoothing_on_everywhere_still_differs_by_method():
    r = compare("bleu", ["a b c d e"], ["a b c x y"], smooth=True, tokenize="whitespace",
                backends=["sacrebleu", "torchmetrics"])
    assert r.confirmed and "bleu_smoothing" in keys(r)
    assert "add-one" in next(c.detail for c in r.causes if c.key == "bleu_smoothing")


def test_per_library_aggregation():
    r = compare("bleu", LONG_R, LONG_P, tokenize="whitespace", backends=["torchmetrics", "nltk"],
                settings={"torchmetrics": {"aggregation": "sentence"}})
    assert r.confirmed and keys(r) == ["text_aggregation"]
    assert "(your setting)" in r.causes[0].detail


def test_lowercase_per_library():
    r = compare("bleu", [s.upper() for s in LONG_R], LONG_P, tokenize="whitespace",
                backends=["torchmetrics", "nltk"], settings={"nltk": {"lowercase": True}})
    assert r.confirmed and keys(r) == ["text_lowercase"]


def test_multiple_references_of_different_counts():
    refs = [[LONG_R[0], "a cat was on the mat"], [LONG_R[1]], [LONG_R[2], "we went to market", "x y z w"]]
    r = compare("bleu", refs, LONG_P, tokenize="whitespace", smooth=False,
                backends=["sacrebleu", "torchmetrics", "nltk", "evaluate"])
    assert r.confirmed and keys(r) == ["text_scale"], str(r)
    assert "1-3 references each" in str(r)


def test_chrf_word_order():
    r = compare("chrf", REFS, PREDS)
    assert r.confirmed and keys(r) == ["chrf_word_order", "text_scale"]
    r = compare("chrf", REFS, PREDS, word_order=2, backends=["sacrebleu", "torchmetrics"])
    assert r.confirmed and keys(r) == ["text_scale"]


def test_rouge_agrees_by_default():
    for m in ("rouge1", "rouge2", "rougeL"):
        assert compare(m, REFS, PREDS).agree


def test_rouge_stemmer():
    r = compare("rouge1", ["the cats are sitting"], ["the cat is sitting"],
                settings={"rouge_score": {"use_stemmer": True}}, backends=["rouge_score", "torchmetrics"])
    assert r.confirmed and keys(r) == ["rouge_stemmer"]


def test_rougelsum_sentence_splitting():
    r = compare("rougeLsum", ["The dog barked at night. The cat sat on a mat."],
                ["The cat sat on the mat. The dog barked loudly at night."])
    assert r.confirmed and keys(r) == ["rougelsum_sentences"]
    assert r.results["torchmetrics"].value == pytest.approx(0.8696, abs=1e-4)
    assert r.results["rouge_score"].value == pytest.approx(0.4348, abs=1e-4)


def test_rougelsum_with_newlines_agrees():
    assert compare("rougeLsum", ["The dog barked at night.\nThe cat sat on a mat."],
                   ["The cat sat on the mat.\nThe dog barked loudly at night."]).agree


def test_non_latin_note():
    r = compare("rouge1", ["猫坐在垫子上了"], ["猫坐在垫子上"])
    assert any("non-Latin" in n for n in r.notes)
    assert all(res.value == 0 for res in r.results.values())


@pytest.mark.parametrize("metric,kwargs,msg", [
    ("rouge1", dict(tokenize="13a"), "takes use_stemmer only"),
    ("chrf", dict(smooth=True), "takes word_order"),
    ("f1", dict(tokenize="13a"), "apply to text metrics"),
    ("f1", dict(aggregation="corpus"), "applies to text metrics"),
    ("bleu", dict(aggregation="per_image"), "'corpus' or 'sentence'"),
    ("bleu", dict(tokenize="moses"), "tokenize must be one of"),
    ("chrf", dict(word_order=3), "word_order must be"),
])
def test_invalid_text_settings(metric, kwargs, msg):
    y = ["a b"] if metric != "f1" else [0, 1]
    with pytest.raises(ValueError, match=msg):
        compare(metric, y, y, **kwargs)


def test_length_mismatch():
    with pytest.raises(ValueError, match="2 references but 1 predictions"):
        compare("bleu", ["a", "b"], ["a"])


def test_testing_helper_with_text():
    assert_metrics_agree("bleu", REFS, PREDS, tokenize="13a", backends=["sacrebleu", "evaluate"],
                         allow=["text_scale"])


def test_aliases():
    assert compare("rouge-L", REFS, PREDS).metric == "rougeL"
    assert compare("sacrebleu", REFS, PREDS, backends=["sacrebleu", "torchmetrics"]).metric == "bleu"


def test_chrf_averaging_on_short_texts():
    r = compare("chrf", ["The cat is sitting on the mat.", "Hello, there."],
                ["The cat sat on the mat.", "Hello there!"], word_order=0,
                backends=["sacrebleu", "torchmetrics"])
    assert r.confirmed and keys(r) == ["chrf_averaging", "text_scale"]


def test_cli_text_with_two_reference_files(tmp_path, capsys):
    from metric_diff.cli import main

    (tmp_path / "hyp.txt").write_text("the cat sat on the mat today\nhello there my friend\n", encoding="utf-8")
    (tmp_path / "r1.txt").write_text("the cat sat on the mat\nhello there friend\n", encoding="utf-8")
    (tmp_path / "r2.txt").write_text("a cat was on the mat\nhi there my friend\n", encoding="utf-8")
    code = main(["bleu", "--true", str(tmp_path / "r1.txt"), "--true", str(tmp_path / "r2.txt"),
                 "--pred", str(tmp_path / "hyp.txt"), "--tokenize", "13a", "--smooth", "false",
                 "--backends", "sacrebleu,torchmetrics"])
    out = capsys.readouterr().out
    assert code == 1 and "2 references each" in out and "[text_scale]" in out


def test_cli_text_line_count_mismatch(tmp_path, capsys):
    from metric_diff.cli import main

    (tmp_path / "hyp.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "ref.txt").write_text("a\n", encoding="utf-8")
    assert main(["bleu", "--true", str(tmp_path / "ref.txt"), "--pred", str(tmp_path / "hyp.txt")]) == 2
    assert "has 1 lines" in capsys.readouterr().err
