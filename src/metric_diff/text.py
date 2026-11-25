"""Text-generation metrics: BLEU, chrF and ROUGE across sacrebleu, nltk, torchmetrics,
Hugging Face evaluate and Google's rouge-score.

Inputs: ``y_true`` = references (one string per sample, or a list of strings for several
references), ``y_pred`` = generated texts (one string per sample).

What usually makes these libraries disagree:

* **scale**: sacrebleu reports BLEU and chrF on 0-100, the others on 0-1 (evaluate's chrF: 0-100);
* **tokenisation** (BLEU): sacrebleu and evaluate use the "13a" tokenizer, which splits off
  punctuation; torchmetrics ``BLEUScore`` and nltk (called with ``str.split``) split on spaces;
* **smoothing** (BLEU): sacrebleu smooths by default ("exp"), the others do not, and when
  smoothing is switched on every library uses a different method;
* **aggregation**: corpus BLEU/chrF (pool n-gram statistics) vs the mean of sentence scores;
* **chrF word order**: torchmetrics ``CHRFScore`` defaults to chrF++ (word n-grams up to 2);
* **nltk short hypotheses**: nltk counts a hypothesis shorter than n words as having one
  (unmatched) n-gram instead of none, which lowers corpus BLEU;
* **ROUGE-Lsum sentence splitting**: torchmetrics splits sentences with nltk, rouge-score and
  evaluate only split on newlines.

``opts`` keys: tokenize, smooth, lowercase, word_order, use_stemmer, aggregation, scale_0_1.
"""

from __future__ import annotations

import re

import numpy as np

from .detectors import Detector
from .inputs import TEXT_METRICS, Case

BLEU_LIKE = ("bleu", "chrf")
ROUGE = ("rouge1", "rouge2", "rougeL", "rougeLsum")
TEXT_BACKENDS = ("sacrebleu", "nltk", "torchmetrics", "evaluate", "rouge_score")

SUPPORTS = {
    "sacrebleu": BLEU_LIKE,
    "nltk": ("bleu",),
    "torchmetrics": TEXT_METRICS,
    "evaluate": TEXT_METRICS,
    "rouge_score": ROUGE,
}

# Each library's behaviour when the user doesn't say otherwise.
DEFAULTS = {
    "sacrebleu": {"tokenize": "13a", "smooth": True, "word_order": 0, "lowercase": False,
                  "aggregation": "corpus"},
    "nltk": {"tokenize": "whitespace", "smooth": False, "lowercase": False, "aggregation": "corpus"},
    "torchmetrics": {"tokenize": "whitespace", "smooth": False, "word_order": 2, "lowercase": False,
                     "use_stemmer": False, "aggregation": "corpus"},
    "evaluate": {"tokenize": "13a", "smooth": False, "word_order": 0, "lowercase": False,
                 "use_stemmer": False, "aggregation": "corpus"},
    "rouge_score": {"use_stemmer": False},
}

# What "smooth=True" means in each library (BLEU).
SMOOTHING = {
    "sacrebleu": "'exp' (Chen & Cherry: halves the precision for each order with no match)",
    "nltk": "method3 (the same 'exp' smoothing as sacrebleu)",
    "torchmetrics": "add-one on n-gram orders 2-4 (Lin & Och)",
    "evaluate": "add-one on all n-gram orders, unigrams included",
}

SCALE_100 = {("sacrebleu", "bleu"), ("sacrebleu", "chrf"), ("evaluate", "chrf")}

# Which settings each metric accepts.
METRIC_KEYS = {
    "bleu": ("tokenize", "smooth", "lowercase", "aggregation"),
    "chrf": ("word_order", "lowercase", "aggregation"),
    **{m: ("use_stemmer",) for m in ROUGE},
}


def default(lib: str, key: str):
    return DEFAULTS.get(lib, {}).get(key)


def effective(case: Case, lib: str, key: str):
    return case.settings.value(lib, key, default(lib, key))


# ---------------------------------------------------------------- inputs


def prepare_text(y_true, y_pred) -> Case:
    preds = _strings(y_pred, "y_pred (generated texts)")
    raw_refs = list(y_true) if not isinstance(y_true, str) else [y_true]
    if len(raw_refs) != len(preds):
        raise ValueError(f"{len(raw_refs)} references but {len(preds)} predictions; "
                         "give one reference (or one list of references) per prediction")
    if not preds:
        raise ValueError("inputs are empty")
    refs = []
    for i, r in enumerate(raw_refs):
        group = [r] if isinstance(r, str) else list(r)
        if not group or not all(isinstance(s, str) for s in group):
            raise ValueError(f"reference {i} must be a string or a non-empty list of strings")
        refs.append(tuple(group))
    yt, yp = np.empty(len(refs), dtype=object), np.empty(len(preds), dtype=object)
    yt[:], yp[:] = refs, preds
    return Case(yt, yp, "text", 0, "text")


def _strings(x, what) -> list[str]:
    if isinstance(x, str):
        x = [x]
    out = list(x)
    if not all(isinstance(s, str) for s in out):
        raise ValueError(f"{what} must be a list of strings")
    return out


def describe(case: Case) -> str:
    k = sorted({len(r) for r in case.y_true})
    refs = (f"{k[0]} reference{'s' if k[0] > 1 else ''} each" if len(k) == 1
            else f"{k[0]}-{k[-1]} references each")
    return f"text, {case.n_samples} generated texts, {refs}"


# ---------------------------------------------------------------- computing


def unsupported(name: str, metric: str, opts: dict) -> str | None:
    if name not in SUPPORTS:
        return f"{name} has no text-generation metrics"
    if metric not in SUPPORTS[name]:
        if name == "nltk" and metric == "chrf":
            return "nltk's chrF uses other definitions (beta=3, whitespace removed); not compared"
        return f"{name} has no {metric}"
    return None


def _tokenizer(name: str):
    if name == "whitespace":
        return lambda s: s
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
    from sacrebleu.tokenizers.tokenizer_char import TokenizerChar
    from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
    from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

    return {"13a": Tokenizer13a, "intl": TokenizerV14International, "char": TokenizerChar,
            "zh": TokenizerZh}[name]()


def _pad(refs: list[tuple]) -> list[list[str]]:
    """Repeat a sample's last reference so every sample has the same number (a duplicate
    reference changes neither BLEU nor chrF), as sacrebleu's reference streams require."""
    k = max(len(r) for r in refs)
    return [list(r) + [r[-1]] * (k - len(r)) for r in refs]


def compute(name: str, metric: str, refs, preds, opts: dict) -> float:
    refs = [tuple(r) for r in refs]
    preds = list(preds)
    if opts.get("lowercase"):
        preds = [p.lower() for p in preds]
        refs = [tuple(s.lower() for s in r) for r in refs]
    if metric in ROUGE:
        return _rouge(name, metric, refs, preds, opts)
    tok = opts.get("tokenize")
    if tok is not None and metric == "bleu":
        t = _tokenizer(tok)
        preds = [t(p) for p in preds]
        refs = [tuple(t(s) for s in r) for r in refs]
    if opts.get("aggregation") == "sentence":
        values = [_one(name, metric, [r], [p], opts, tok, sentence=True) for r, p in zip(refs, preds)]
        value = float(np.mean(values))
    else:
        value = _one(name, metric, refs, preds, opts, tok, sentence=False)
    if opts.get("scale_0_1") and (name, metric) in SCALE_100:
        value /= 100.0
    return value


def _one(name, metric, refs, preds, opts, tok, sentence):
    smooth = opts.get("smooth", default(name, "smooth"))
    if name == "sacrebleu":
        import logging

        import sacrebleu

        logging.getLogger("sacrebleu").setLevel(logging.ERROR)  # "enable effective_order" advice

        if metric == "chrf":
            from sacrebleu.metrics import CHRF

            chrf = CHRF(word_order=opts.get("word_order", 0), eps_smoothing=bool(opts.get("chrf_eps")))
            if sentence:
                return chrf.sentence_score(preds[0], list(refs[0])).score
            return chrf.corpus_score(preds, list(zip(*_pad(refs)))).score
        kw = {"tokenize": "none" if tok is not None else "13a",
              "smooth_method": "exp" if smooth else "none"}
        if sentence:
            return sacrebleu.sentence_bleu(preds[0], list(refs[0]), use_effective_order=bool(smooth),
                                           **kw).score
        return sacrebleu.corpus_bleu(preds, list(zip(*_pad(refs))), **kw).score

    if name == "nltk":
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

        fn = SmoothingFunction().method3 if smooth else None
        hyp = [p.split() for p in preds]
        ref = [[s.split() for s in r] for r in refs]
        if sentence:
            return float(sentence_bleu(ref[0], hyp[0], smoothing_function=fn))
        return float(corpus_bleu(ref, hyp, smoothing_function=fn))

    if name == "torchmetrics":
        from torchmetrics.functional.text import bleu_score, chrf_score

        if metric == "chrf":
            wo = opts.get("word_order", 2)
            return float(chrf_score(preds, [list(r) for r in refs], n_word_order=wo))
        return float(bleu_score(preds, [list(r) for r in refs], smooth=bool(smooth)))

    if name == "evaluate":
        from .backends import _load_evaluate

        if metric == "chrf":
            out = _load_evaluate("chrf").compute(predictions=preds, references=[list(r) for r in _pad(refs)],
                                                 word_order=opts.get("word_order", 0),
                                                 eps_smoothing=bool(opts.get("chrf_eps")))
            return float(out["score"])
        kw = {"tokenizer": str.split} if tok is not None else {}
        out = _load_evaluate("bleu").compute(predictions=preds, references=[list(r) for r in refs],
                                             smooth=bool(smooth), **kw)
        return float(out["bleu"])
    raise ValueError(f"{name} has no {metric}")


def _rouge(name, metric, refs, preds, opts):
    stem = bool(opts.get("use_stemmer", False))
    if name == "rouge_score":
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=stem)
        return float(np.mean([scorer.score_multi(list(r), p)[metric].fmeasure for r, p in zip(refs, preds)]))
    if name == "torchmetrics":
        from torchmetrics.functional.text import rouge_score

        out = rouge_score(preds, [list(r) for r in refs], rouge_keys=metric, use_stemmer=stem)
        return float(out[f"{metric}_fmeasure"])
    if name == "evaluate":
        from .backends import _load_evaluate

        out = _load_evaluate("rouge").compute(predictions=preds, references=[list(r) for r in refs],
                                              rouge_types=[metric], use_stemmer=stem)
        return float(out[metric])
    raise ValueError(f"{name} has no {metric}")


# ---------------------------------------------------------------- detectors


class _Setting(Detector):
    """A setting whose effective value (user setting, else library default) differs."""

    setting = ""
    metrics: tuple = ()

    def _libs(self, case, metric):
        return [n for n in case.backends if metric in SUPPORTS.get(n, ())]

    def values(self, case, metric) -> dict:
        return {n: effective(case, n, self.setting) for n in self._libs(case, metric)
                if effective(case, n, self.setting) is not None}

    def relevant(self, case, metric):
        return case.task == "text" and metric in self.metrics and len(set(self.values(case, metric).values())) > 1

    def target(self, case, metric):
        common = case.settings.common.get(self.setting)
        if common is not None:
            return common
        vals = self.values(case, metric)
        for ref in ("sacrebleu", "rouge_score"):
            if ref in vals:
                return vals[ref]
        return next(iter(vals.values()))

    def options(self, case, metric):
        return {self.setting: self.target(case, metric)}

    def affected(self, case, metric, names):
        target = self.options(case, metric)[self.setting]
        return {n for n, v in self.values(case, metric).items() if v != target} & set(names)

    def _listing(self, case, metric, fmt=repr):
        return "; ".join(f"{n}: {fmt(v)}" + (" (your setting)" if case.settings.value(n, self.setting) is not None
                                             else "") for n, v in self.values(case, metric).items())


class TextScale(Detector):
    key = "text_scale"
    title = "Different scales: 0-100 vs 0-1"

    def relevant(self, case, metric):
        if case.task != "text" or metric not in BLEU_LIKE:
            return False
        libs = [n for n in case.backends if metric in SUPPORTS.get(n, ())]
        hundred = [n for n in libs if (n, metric) in SCALE_100]
        return bool(hundred) and len(hundred) < len(libs)

    def options(self, case, metric):
        return {"scale_0_1": True}

    def affected(self, case, metric, names):
        return {n for n in names if (n, metric) in SCALE_100}

    def explain(self, case, metric, backends):
        hundred = [n for n in backends if (n, metric) in SCALE_100]
        return (f"{', '.join(hundred)} report{'s' if len(hundred) == 1 else ''} {metric} on a 0-100 scale; "
                "the other libraries on 0-1. The scores are the same once divided by 100.",
                "Divide by 100 (or multiply the others) before comparing or logging side by side.")


class BleuTokenize(_Setting):
    key = "bleu_tokenize"
    title = "Different tokenisation"
    setting = "tokenize"
    metrics = ("bleu",)
    reference = "Post 2018, 'A Call for Clarity in Reporting BLEU Scores' (sacreBLEU)"

    def explain(self, case, metric, backends):
        return ("BLEU counts matching word n-grams, so how text is split into words changes the score. "
                "The 13a tokenizer (sacrebleu, evaluate's bleu) splits off punctuation: 'mat.' becomes "
                "'mat .'; torchmetrics BLEUScore and nltk (called on text.split()) split on spaces only, "
                f"so 'mat.' and 'mat' don't match. In use here: {self._listing(case, metric)}.",
                "Use the same tokenizer everywhere: tokenize='13a' in sacrebleu, torchmetrics "
                "SacreBLEUScore(tokenize='13a') instead of BLEUScore, and sacrebleu's tokenizer before "
                "nltk. Report BLEU with sacrebleu's signature so others can reproduce it.")


class BleuSmoothing(_Setting):
    key = "bleu_smoothing"
    title = "Different smoothing"
    setting = "smooth"
    metrics = ("bleu",)
    reference = "Chen & Cherry 2014, 'A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU'"

    def relevant(self, case, metric):
        if case.task != "text" or metric != "bleu":
            return False
        vals = self.values(case, metric)
        if len(set(vals.values())) > 1:
            return True
        # All on: still different unless they use the same method (sacrebleu and nltk do).
        methods = {"exp" if n in ("sacrebleu", "nltk") else n for n, v in vals.items() if v}
        return len(methods) > 1

    def options(self, case, metric):
        return {"smooth": False}

    def affected(self, case, metric, names):
        return {n for n, v in self.values(case, metric).items() if v} & set(names)

    def explain(self, case, metric, backends):
        on = [n for n, v in self.values(case, metric).items() if v]
        methods = "; ".join(f"{n}: {SMOOTHING[n]}" for n in on if n in SMOOTHING)
        return ("Smoothing changes BLEU when some n-gram order has few or no matches (short texts, "
                "sentence-level BLEU). sacrebleu smooths by default; nltk, torchmetrics and evaluate do "
                f"not, and each library smooths differently when asked. Smoothing in use: {methods or 'none'}. "
                f"Settings: {self._listing(case, metric)}.",
                "Switch smoothing off everywhere for corpus BLEU (sacrebleu: smooth_method='none'), or "
                "use one library for sentence-level BLEU.")


class TextAggregation(_Setting):
    key = "text_aggregation"
    title = "Corpus score vs mean of sentence scores"
    setting = "aggregation"
    metrics = BLEU_LIKE

    def explain(self, case, metric, backends):
        return (f"Corpus {metric} pools the n-gram counts of all texts and computes one score; averaging "
                f"sentence-level {metric} gives a different, usually much lower, number for BLEU (one "
                f"sentence without a 4-gram match scores 0). In use: {self._listing(case, metric)}.",
                "Report corpus-level scores (sacrebleu.corpus_bleu, BLEUScore over all texts), not the mean "
                "of per-sentence scores, unless you need per-sentence values.")


class ChrfWordOrder(_Setting):
    key = "chrf_word_order"
    title = "chrF vs chrF++ (word n-grams)"
    setting = "word_order"
    metrics = ("chrf",)
    reference = "Popovic 2017, 'chrF++: words helping character n-grams'"

    def explain(self, case, metric, backends):
        return ("chrF averages character n-gram F-scores; chrF++ also adds word unigrams and bigrams "
                "(word_order=2). torchmetrics CHRFScore defaults to n_word_order=2, i.e. chrF++, while "
                f"sacrebleu and evaluate default to plain chrF. In use: {self._listing(case, metric)}.",
                "Set it explicitly: CHRFScore(n_word_order=0) for chrF, or word_order=2 in sacrebleu / "
                "evaluate for chrF++, and say which one you report.")


class ChrfAveraging(Detector):
    key = "chrf_averaging"
    title = "chrF: averaging over n-gram orders differs"
    affects = frozenset({"sacrebleu", "evaluate"})
    reference = "sacrebleu CHRF(eps_smoothing=...) documentation"

    def relevant(self, case, metric):
        libs = set(case.backends)
        return (case.task == "text" and metric == "chrf" and "torchmetrics" in libs
                and bool(libs & {"sacrebleu", "evaluate"}))

    def options(self, case, metric):
        return {"chrf_eps": True}

    def explain(self, case, metric, backends):
        return ("sacrebleu (and evaluate, which wraps it) by default averages precision and recall over "
                "the n-gram orders first and then computes one F-score (eps_smoothing=False). "
                "torchmetrics computes an F-score per order and averages those, as the original chrF++.py "
                "and nltk do (sacrebleu's eps_smoothing=True). The gap is small on long texts and larger "
                "on short ones.",
                "Pass eps_smoothing=True to sacrebleu's CHRF (and evaluate's chrf) to match torchmetrics, "
                "or compute chrF with one library only.")


class TextLowercase(_Setting):
    key = "text_lowercase"
    title = "Case-sensitive vs lowercased"
    setting = "lowercase"
    metrics = BLEU_LIKE

    def explain(self, case, metric, backends):
        return (f"With lowercasing, 'The' and 'the' match; without it they don't. In use: "
                f"{self._listing(case, metric)}.",
                "Lowercase in every library or in none (sacrebleu lowercase=True / -lc, torchmetrics "
                "lowercase=True).")


class RougeStemmer(_Setting):
    key = "rouge_stemmer"
    title = "Stemming on in one library, off in another"
    setting = "use_stemmer"
    metrics = ROUGE

    def explain(self, case, metric, backends):
        return ("With use_stemmer=True, 'sitting' and 'sit' match (Porter stemmer). In use: "
                f"{self._listing(case, metric)}.",
                "Pass the same use_stemmer to every library. Most papers use use_stemmer=True.")


class NltkShortHypotheses(Detector):
    key = "nltk_short_hypotheses"
    title = "nltk counts texts shorter than 4 words differently"
    requires = "nltk"
    affects = frozenset({"nltk"})
    reference = "nltk.translate.bleu_score.modified_precision: denominator max(1, count)"

    def _tokens(self, case, text):
        tok = case.settings.value("nltk", "tokenize")
        return (_tokenizer(tok)(text) if tok else text).split()

    def relevant(self, case, metric):
        return (case.task == "text" and metric == "bleu"
                and effective(case, "nltk", "aggregation") == "corpus"
                and any(len(self._tokens(case, p)) < 4 for p in case.y_pred))

    def transform(self, y_true, y_pred, case):
        keep = np.array([len(self._tokens(case, p)) >= 4 for p in y_pred], dtype=bool)
        if not keep.any():
            return y_true, y_pred
        return y_true[keep], y_pred[keep]

    def explain(self, case, metric, backends):
        short = sum(len(self._tokens(case, p)) < 4 for p in case.y_pred)
        return (f"{short} of {case.n_samples} generated texts have fewer than 4 words. For such a text, "
                "nltk's corpus_bleu counts one possible n-gram (instead of none) for each order it is too "
                "short for, so corpus BLEU comes out lower than in sacrebleu, torchmetrics or evaluate. "
                "Without these texts the libraries agree.",
                "Use sacrebleu for corpus BLEU, or remember that nltk penalises very short outputs more. "
                "The difference shrinks as texts get longer.")


class RougeLsumSentences(Detector):
    key = "rougelsum_sentences"
    title = "ROUGE-Lsum: sentences split by torchmetrics, not by the others"
    requires = "torchmetrics"
    affects = frozenset({"torchmetrics"})
    reference = "google-research/rouge rouge_scorer: rougeLsum splits on newlines"

    def relevant(self, case, metric):
        if case.task != "text" or metric != "rougeLsum":
            return False
        texts = list(case.y_pred) + [s for r in case.y_true for s in r]
        return any("\n" not in t and len(_sentences(t)) > 1 for t in texts)

    def transform(self, y_true, y_pred, case):
        yt, yp = np.empty(len(y_true), dtype=object), np.empty(len(y_pred), dtype=object)
        yt[:] = [tuple(_newlines(s) for s in r) for r in y_true]
        yp[:] = [_newlines(p) for p in y_pred]
        return yt, yp

    def explain(self, case, metric, backends):
        return ("ROUGE-Lsum computes ROUGE-L per sentence and needs to know where sentences end. "
                "rouge-score and evaluate only split on newlines, so a summary written as one line is a "
                "single 'sentence' and rougeLsum equals rougeL. torchmetrics splits sentences itself with "
                "nltk. With one sentence per line, the libraries agree.",
                "Put each sentence on its own line before scoring (e.g. '\\n'.join(nltk.sent_tokenize(text))), "
                "as the CNN/DailyMail summarisation papers do.")


def _sentences(text: str) -> list[str]:
    try:
        import nltk

        return nltk.sent_tokenize(text)
    except Exception:  # punkt missing: a rough split is enough to decide relevance
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _newlines(text: str) -> str:
    return text if "\n" in text else "\n".join(_sentences(text))


TEXT_DETECTORS = [TextScale(), BleuTokenize(), BleuSmoothing(), TextAggregation(), ChrfWordOrder(),
                  ChrfAveraging(), TextLowercase(), RougeStemmer(), NltkShortHypotheses(), RougeLsumSentences()]


# ---------------------------------------------------------------- notes

_NON_LATIN = re.compile(r"[^\x00-ɏ\s]")


def notes(case: Case, metric: str) -> list[str]:
    out = []
    texts = list(case.y_pred) + [s for r in case.y_true for s in r]
    non_latin = sum(bool(_NON_LATIN.search(t)) for t in texts)
    if metric in ROUGE and non_latin:
        out.append("some texts contain non-Latin characters: rouge-score, torchmetrics and evaluate keep "
                   "only a-z and 0-9 by default, so those characters are dropped and scores can be 0 "
                   "for every library. Pass your own tokenizer (rouge-score / evaluate: tokenizer=...).")
    if metric == "bleu" and non_latin and not case.settings.values("tokenize"):
        out.append("some texts contain non-Latin characters: for Chinese use tokenize='zh', for Japanese "
                   "or Thai tokenize='char' (or a language-specific tokenizer); 13a and whitespace "
                   "tokenisation treat a whole unspaced sentence as one word.")
    if metric == "bleu" and "nltk" in case.backends:
        out.append("nltk has no tokenizer argument: metric-diff splits on whitespace (text.split()), "
                   "as most users do.")
    return out
