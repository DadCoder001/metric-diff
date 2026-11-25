"""Ranking and retrieval metrics: NDCG, MAP, MRR, precision@k, recall@k and hit rate across
scikit-learn, torchmetrics, torcheval and pytrec_eval (the Python binding of trec_eval, the
reference used by TREC, MS MARCO and BEIR).

Inputs: one row per query. ``y_true`` = relevance of each candidate (0 = not relevant; graded
values such as 0-3 for NDCG), ``y_pred`` = the model's score for each candidate. A 2-D array
when every query has the same number of candidates, otherwise a list of 1-D arrays.

What usually makes these libraries disagree:

* **tied scores**: sklearn averages over the possible orders of tied candidates (NDCG) or
  treats a tie as one threshold (average precision); torchmetrics keeps the order its sort
  returns; trec_eval breaks ties by document id, in reverse order;
* **MAP@k**: trec_eval (map_cut) divides by every relevant document of the query, torchmetrics
  by the relevant documents it found in the top k;
* **queries without any relevant document**: counted as 0 (the default everywhere), skipped
  (torchmetrics ``empty_target_action="skip"``, common in evaluation scripts) or counted as 1;
* **short candidate lists for precision@k**: divide by k (default) or by the list length
  (torchmetrics ``adaptive_k``, torcheval ``limit_k_to_size``).

``opts`` keys: top_k, empty_queries, adaptive_k.
"""

from __future__ import annotations

import numpy as np

from .detectors import Detector
from .inputs import RETRIEVAL_METRICS as METRICS
from .inputs import Case

NEEDS_K = ("precision_at_k", "recall_at_k", "hit_rate")
GRADED = ("ndcg",)  # metrics that use graded relevance; the others treat relevance > 0 as relevant

SUPPORTS = {
    "sklearn": ("ndcg", "map"),
    "torchmetrics": METRICS,
    "torcheval": ("precision_at_k",),
    "pytrec_eval": METRICS,
}

DEFAULTS = {lib: {"empty_queries": "zero", "adaptive_k": False} for lib in SUPPORTS}

TREC = {"ndcg": "ndcg", "map": "map", "mrr": "recip_rank", "precision_at_k": "P",
        "recall_at_k": "recall", "hit_rate": "success"}


def default(lib: str, key: str):
    return DEFAULTS.get(lib, {}).get(key)


def effective(case: Case, lib: str, key: str):
    return case.settings.value(lib, key, default(lib, key))


# ---------------------------------------------------------------- inputs


def _rows(x, what) -> list[np.ndarray]:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, np.ndarray):
        if x.ndim != 2:
            raise ValueError(f"{what} must be 2-D (n_queries, n_candidates) or a list of 1-D arrays, "
                             f"got shape {x.shape}")
        rows = list(x)
    else:
        rows = [np.asarray(r) for r in x]
    if not rows:
        raise ValueError("inputs are empty")
    out = []
    for i, r in enumerate(rows):
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 1 or r.size == 0:
            raise ValueError(f"{what}: query {i} must be a non-empty 1-D list of numbers")
        out.append(r)
    return out


def prepare_ranking(y_true, y_pred) -> Case:
    rel, scores = _rows(y_true, "y_true (relevance)"), _rows(y_pred, "y_pred (scores)")
    if len(rel) != len(scores):
        raise ValueError(f"y_true has {len(rel)} queries but y_pred has {len(scores)}")
    for i, (r, s) in enumerate(zip(rel, scores)):
        if r.shape != s.shape:
            raise ValueError(f"query {i}: {r.size} relevance values but {s.size} scores")
        if (r < 0).any() or not np.all(np.mod(r, 1) == 0):
            raise ValueError(f"query {i}: relevance must be non-negative integers (0 = not relevant)")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"query {i}: scores must be finite")
    yt, yp = np.empty(len(rel), dtype=object), np.empty(len(rel), dtype=object)
    yt[:], yp[:] = rel, scores
    graded = any((r > 1).any() for r in rel)
    return Case(yt, yp, "ranking", 0, "graded" if graded else "binary")


def describe(case: Case) -> str:
    sizes = sorted({r.size for r in case.y_true})
    cand = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
    rel = (f"graded relevance 0-{int(max(r.max() for r in case.y_true))}" if case.pred_kind == "graded"
           else "binary relevance")
    return f"ranking, {case.n_samples} queries, {cand} candidates each, {rel}"


# ---------------------------------------------------------------- computing


def unsupported(name: str, metric: str, opts: dict) -> str | None:
    if name not in SUPPORTS:
        return f"{name} has no ranking metrics"
    if metric not in SUPPORTS[name]:
        return f"{name} has no {metric}"
    k = opts.get("top_k")
    if name == "sklearn" and metric == "map" and k is not None:
        return "sklearn's average_precision_score has no cut-off k"
    if name == "pytrec_eval" and metric == "mrr" and k is not None:
        return "trec_eval's recip_rank has no cut-off k"
    if name == "pytrec_eval" and opts.get("adaptive_k"):
        return "trec_eval always divides precision@k by k"
    if name == "torcheval" and opts.get("empty_queries") not in (None, "zero"):
        return "torcheval has no option for queries without relevant documents"
    return None


def compute(name: str, metric: str, rel, scores, opts: dict) -> float:
    rel, scores = list(rel), list(scores)
    k = opts.get("top_k")
    if metric in NEEDS_K and k is None:
        raise ValueError(f"{metric} needs top_k")
    if name == "torchmetrics":
        return _torchmetrics(metric, rel, scores, opts)
    if name == "pytrec_eval":
        return _trec(metric, rel, scores, opts)
    policy = opts.get("empty_queries", "zero")
    values = []
    for r, s in zip(rel, scores):
        if not (r > 0).any() and metric != "precision_at_k":
            if policy == "skip":
                continue
            values.append(1.0 if policy == "one" else 0.0)
            continue
        values.append(_one(name, metric, r, s, opts))
    if not values:
        raise ValueError("no query left to score")
    return float(np.mean(values))


def _one(name, metric, r, s, opts):
    k = opts.get("top_k")
    if name == "sklearn":
        from sklearn.metrics import average_precision_score, ndcg_score

        if metric == "ndcg":
            if r.size < 2:
                raise ValueError("sklearn's ndcg_score needs at least 2 candidates per query")
            return float(ndcg_score([r], [s], k=k))
        return float(average_precision_score((r > 0).astype(int), s))
    if name == "torcheval":
        import torch
        from torcheval.metrics.functional import retrieval_precision

        return float(retrieval_precision(torch.tensor(s), torch.tensor((r > 0).astype(np.int64)), k=k,
                                         limit_k_to_size=bool(opts.get("adaptive_k"))))
    raise ValueError(f"{name} has no {metric}")


def _torchmetrics(metric, rel, scores, opts):
    import torch
    from torchmetrics import retrieval as R

    cls = {"ndcg": R.RetrievalNormalizedDCG, "map": R.RetrievalMAP, "mrr": R.RetrievalMRR,
           "precision_at_k": R.RetrievalPrecision, "recall_at_k": R.RetrievalRecall,
           "hit_rate": R.RetrievalHitRate}[metric]
    action = {"zero": "neg", "skip": "skip", "one": "pos"}[opts.get("empty_queries", "zero")]
    kw = {"empty_target_action": action}
    if opts.get("top_k") is not None:
        kw["top_k"] = int(opts["top_k"])
    if metric == "precision_at_k":
        kw["adaptive_k"] = bool(opts.get("adaptive_k"))
    m = cls(**kw)
    idx = torch.cat([torch.full((r.size,), i, dtype=torch.long) for i, r in enumerate(rel)])
    preds = torch.tensor(np.concatenate(scores), dtype=torch.float32)
    target = np.concatenate(rel)
    target = torch.tensor(target.astype(np.int64)) if metric in GRADED else torch.tensor(target > 0)
    return float(m(preds, target, indexes=idx))


def _trec(metric, rel, scores, opts):
    import pytrec_eval

    k = opts.get("top_k")
    measure = TREC[metric]
    if metric == "ndcg" and k is not None:
        measure = f"ndcg_cut.{k}"
    elif metric == "map" and k is not None:
        measure = f"map_cut.{k}"
    elif metric in NEEDS_K:
        measure = f"{measure}.{k}"
    policy = opts.get("empty_queries", "zero")
    qrels, run, empty = {}, {}, []
    for i, (r, s) in enumerate(zip(rel, scores)):
        q = f"q{i}"
        if not (r > 0).any() and metric != "precision_at_k" and policy != "zero":
            empty.append(q)
            continue
        qrels[q] = {f"d{j:05d}": int(v) for j, v in enumerate(r)}
        run[q] = {f"d{j:05d}": float(v) for j, v in enumerate(s)}
    if not qrels:
        raise ValueError("no query left to score")
    result = pytrec_eval.RelevanceEvaluator(qrels, {measure}).evaluate(run)
    values = [next(iter(v.values())) for v in result.values()]
    if policy == "one":
        values += [1.0] * len(empty)
    return float(np.mean(values))


# ---------------------------------------------------------------- detectors


def _libs(case, metric):
    return [n for n in case.backends if metric in SUPPORTS.get(n, ())]


def _has_ties(r, s) -> bool:
    """Tied scores among candidates of different relevance (other ties change nothing)."""
    for v in np.unique(s):
        same = r[s == v]
        if same.size > 1 and np.unique(same).size > 1:
            return True
    return False


def _break_ties(s: np.ndarray) -> np.ndarray:
    """Lower each tied score slightly, in listed order, so every library sees one ranking.
    The steps are a fraction of the smallest gap between distinct scores (and large enough for
    float32), so no candidate moves past another score, and signs are kept."""
    u = np.unique(s)
    gap = float(np.min(np.diff(u))) if u.size > 1 else max(abs(float(u[0])), 1.0)
    step = gap / (s.size + 1)
    out = s.astype(np.float64).copy()
    for v in u:
        pos = np.flatnonzero(s == v)
        out[pos] = v - step * np.arange(pos.size)
    return out


class RankingTies(Detector):
    key = "ranking_ties"
    title = "Tied scores are ordered differently"
    reference = "trec_eval orders ties by document id; sklearn ndcg_score(ignore_ties=False)"

    def relevant(self, case, metric):
        return case.task == "ranking" and any(_has_ties(r, s) for r, s in zip(case.y_true, case.y_pred))

    def transform(self, y_true, y_pred, case):
        yp = np.empty(len(y_pred), dtype=object)
        yp[:] = [_break_ties(s) for s in y_pred]
        return y_true, yp

    def explain(self, case, metric, backends):
        n = sum(_has_ties(r, s) for r, s in zip(case.y_true, case.y_pred))
        how = {
            "sklearn": "averages over every order of the tied candidates (NDCG) or treats them as one "
                       "threshold (average precision)",
            "torchmetrics": "keeps the order its sort returns, which is not specified for ties",
            "torcheval": "keeps the order torch.topk returns",
            "pytrec_eval": "breaks ties by document id, in reverse alphabetical order (trec_eval)",
        }
        libs = "; ".join(f"{b} {how[b]}" for b in backends if b in how)
        return (f"{n} of {case.n_samples} queries have tied scores between relevant and non-relevant "
                f"candidates, and the libraries order ties differently: {libs}. Once the ties are broken "
                "the same way, the libraries agree.",
                "Break ties yourself before evaluating (e.g. add a tiny, deterministic secondary key), or "
                "check why your model gives identical scores (rounding, saturated sigmoid, fp16).")


class TorchmetricsNonPositive(Detector):
    key = "torchmetrics_nonpositive_scores"
    title = "torchmetrics ignores candidates scored 0 or below"
    requires = "torchmetrics"
    affects = frozenset({"torchmetrics"})
    reference = "Lightning-AI/torchmetrics#3518"
    METRICS = ("map", "mrr", "precision_at_k", "recall_at_k")

    def relevant(self, case, metric):
        return (case.task == "ranking" and metric in self.METRICS
                and any(((s <= 0) & (r > 0)).any() for r, s in zip(case.y_true, case.y_pred)))

    def transform(self, y_true, y_pred, case):
        yp = np.empty(len(y_pred), dtype=object)
        yp[:] = [s - s.min() + 1.0 for s in y_pred]  # same ranking, all scores positive
        return y_true, yp

    def explain(self, case, metric, backends):
        n = sum(int(((s <= 0) & (r > 0)).sum()) for r, s in zip(case.y_true, case.y_pred))
        return (f"{n} relevant candidates have a score of 0 or below. torchmetrics' MAP, MRR, precision "
                "and recall treat such candidates as not retrieved (they are dropped before ranking), "
                "so raw logits, cosine similarities or log-probabilities give lower values than in the "
                "other libraries, which only use the order of the scores. With all scores shifted above "
                "0 (same ranking), the libraries agree.",
                "Pass torchmetrics positive scores with the same order, e.g. torch.sigmoid(logits) or "
                "scores - scores.min() + 1 per query. NDCG and hit rate are not affected.")


class MapCutoff(Detector):
    key = "map_cutoff"
    title = "MAP@k: different denominators"
    reference = "trec_eval map_cut; torchmetrics RetrievalMAP(top_k)"

    def relevant(self, case, metric):
        return (case.task == "ranking" and metric == "map" and bool(case.settings.values("top_k"))
                and {"torchmetrics", "pytrec_eval"} <= set(case.backends))

    def options(self, case, metric):
        return {"top_k": None}

    def explain(self, case, metric, backends):
        return ("MAP@k sums the precision at each relevant document in the top k, then divides. "
                "trec_eval (map_cut.k) divides by every relevant document the query has, so relevant "
                "documents below rank k count as misses. torchmetrics RetrievalMAP(top_k=k) divides by the "
                "relevant documents found in the top k only, which gives higher values. Without the "
                "cut-off (full MAP) the libraries agree.",
                "Say which definition you report; for comparison with TREC/BEIR results use trec_eval's "
                "(pytrec_eval map_cut.k).")


class _Setting(Detector):
    setting = ""
    metrics: tuple = ()

    def values(self, case, metric) -> dict:
        return {n: effective(case, n, self.setting) for n in _libs(case, metric)}

    def relevant(self, case, metric):
        return (case.task == "ranking" and metric in self.metrics and self.applies(case)
                and len(set(self.values(case, metric).values())) > 1)

    def applies(self, case) -> bool:
        return True

    def options(self, case, metric):
        common = case.settings.common.get(self.setting)
        return {self.setting: common if common is not None else default("pytrec_eval", self.setting)}

    def affected(self, case, metric, names):
        target = self.options(case, metric)[self.setting]
        return {n for n, v in self.values(case, metric).items() if v != target} & set(names)

    def _listing(self, case, metric):
        return "; ".join(f"{n}: {v!r}" + (" (your setting)" if case.settings.value(n, self.setting) is not None
                                         else "") for n, v in self.values(case, metric).items())


class EmptyQueries(_Setting):
    key = "empty_queries"
    title = "Queries without relevant documents are scored differently"
    setting = "empty_queries"
    metrics = METRICS

    def applies(self, case):
        return any(not (r > 0).any() for r in case.y_true)

    def explain(self, case, metric, backends):
        n = sum(not (r > 0).any() for r in case.y_true)
        return (f"{n} of {case.n_samples} queries have no relevant candidate, so {metric} is undefined for "
                "them. Counting them as 0 lowers the mean, skipping them raises it. In use: "
                f"{self._listing(case, metric)}.",
                "Choose one policy and use it everywhere (torchmetrics empty_target_action='neg'|'skip'|'pos'; "
                "for trec_eval, leave such queries out of the qrels to skip them), and report it.")


class AdaptiveK(_Setting):
    key = "adaptive_k"
    title = "precision@k on short lists: divide by k or by the list length"
    setting = "adaptive_k"
    metrics = ("precision_at_k",)

    def applies(self, case):
        ks = case.settings.values("top_k")
        return bool(ks) and any(r.size < max(ks) for r in case.y_true)

    def explain(self, case, metric, backends):
        return ("Some queries have fewer candidates than k. Dividing by k (default everywhere, and trec_eval's "
                "only option) counts the missing positions as misses; dividing by the list length "
                "(torchmetrics adaptive_k=True, torcheval limit_k_to_size=True) does not. In use: "
                f"{self._listing(case, metric)}.",
                "Use the same option in every library, and say which one you report.")


RANKING_DETECTORS = [TorchmetricsNonPositive(), RankingTies(), MapCutoff(), EmptyQueries(), AdaptiveK()]


def notes(case: Case, metric: str) -> list[str]:
    out = []
    if case.pred_kind == "graded" and metric not in GRADED:
        out.append(f"relevance is graded; {metric} treats every relevance > 0 as relevant "
                   "(trec_eval: relevance_level=1).")
    if metric == "ndcg" and case.pred_kind == "graded":
        out.append("NDCG here uses linear gain (relevance), as sklearn, torchmetrics and trec_eval do; "
                   "code using exponential gain (2^relevance - 1, Burges et al.) gives different values.")
    return out
