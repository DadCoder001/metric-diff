"""Ranking / retrieval metrics across sklearn, torchmetrics, torcheval and pytrec_eval."""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")
pytest.importorskip("pytrec_eval")

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.testing import assert_metrics_agree  # noqa: E402

REL = [[3, 2, 0, 1, 0], [0, 1, 0, 0, 1], [2, 0, 0, 1, 0]]
SC = [[0.9, 0.8, 0.7, 0.2, 0.1], [0.5, 0.4, 0.9, 0.3, 0.2], [0.3, 0.6, 0.1, 0.8, 0.2]]
TIES = [[0.5, 0.5, 0.5, 0.2, 0.1], [0.5, 0.5, 0.9, 0.5, 0.2], [0.3, 0.3, 0.3, 0.3, 0.2]]


def keys(r):
    return sorted(c.key for c in r.causes)


@pytest.mark.parametrize("metric,kw,value", [
    ("ndcg", {}, 0.7631), ("ndcg", {"top_k": 3}, 0.6539), ("map", {}, 0.7056), ("mrr", {}, 0.7778),
    ("precision_at_k", {"top_k": 3}, 0.5556), ("recall_at_k", {"top_k": 3}, 0.7222),
    ("hit_rate", {"top_k": 2}, 0.6667),
])
def test_all_libraries_agree_without_ties(metric, kw, value):
    r = compare(metric, REL, SC, **kw)
    assert r.agree, str(r)
    assert len([x for x in r.results.values() if x.ok]) >= 2
    assert next(iter(r.results.values())).value == pytest.approx(value, abs=1e-4)


def test_ragged_queries():
    rel = [[1, 0, 1], [0, 0, 1, 0, 1, 0], [1, 0]]
    sc = [[0.9, 0.2, 0.4], [0.1, 0.5, 0.7, 0.3, 0.6, 0.2], [0.3, 0.8]]
    r = compare("map", rel, sc)
    assert r.agree and "2-6 candidates" in str(r)


@pytest.mark.parametrize("metric", ["map", "ndcg", "mrr"])
def test_ties_explained(metric):
    r = compare(metric, REL, TIES)
    assert r.status == "mismatch" and r.confirmed and keys(r) == ["ranking_ties"]


def test_map_ties_differ_by_library():
    r = compare("map", REL, TIES)
    v = {k: round(x.value, 4) for k, x in r.results.items()}
    assert v["sklearn"] == 0.5065 and v["pytrec_eval"] == 0.5852
    # torchmetrics keeps its sort order for ties, which differs by platform
    # (0.6778 on Windows, 0.5852 on Linux, 0.7056 on macOS in CI).
    assert v["torchmetrics"] in (0.5852, 0.6778, 0.7056)


def test_ties_among_equally_relevant_candidates_do_not_count():
    r = compare("map", [[1, 1, 0]], [[0.5, 0.5, 0.1]])
    assert r.agree


@pytest.mark.parametrize("metric", ["map", "mrr", ("precision_at_k", 4), ("recall_at_k", 4)])
def test_torchmetrics_drops_nonpositive_scores(metric):
    metric, k = metric if isinstance(metric, tuple) else (metric, None)
    logits = [[x * 4 - 2 for x in s] for s in SC]
    r = compare(metric, REL, logits, top_k=k)
    assert r.confirmed and keys(r) == ["torchmetrics_nonpositive_scores"]
    assert "#3518" in r.causes[0].reference


def test_nonpositive_scores_do_not_affect_ndcg():
    assert compare("ndcg", REL, [[x * 4 - 2 for x in s] for s in SC]).agree


def test_map_at_k_denominators():
    r = compare("map", REL, SC, top_k=3)
    assert r.results["sklearn"].skipped
    assert r.results["torchmetrics"].value == pytest.approx(0.7222, abs=1e-4)
    assert r.results["pytrec_eval"].value == pytest.approx(0.5556, abs=1e-4)
    assert r.confirmed and keys(r) == ["map_cutoff"]


def test_empty_queries_policy():
    rel, sc = REL + [[0, 0, 0, 0, 0]], SC + [[0.1, 0.2, 0.3, 0.4, 0.5]]
    assert compare("ndcg", rel, sc).agree  # all count it as 0 by default
    r = compare("ndcg", rel, sc, settings={"torchmetrics": {"empty_queries": "skip"}})
    assert r.confirmed and keys(r) == ["empty_queries"]
    assert compare("ndcg", rel, sc, empty_queries="skip").agree
    r = compare("ndcg", rel, sc, empty_queries="one")
    assert r.agree and r.results["pytrec_eval"].value == pytest.approx((0.7631 * 3 + 1) / 4, abs=1e-4)


def test_adaptive_k():
    rel, sc = [[1, 0, 1], [1, 0, 0, 0, 1]], [[0.9, 0.8, 0.7], [0.9, 0.1, 0.2, 0.3, 0.8]]
    r = compare("precision_at_k", rel, sc, top_k=5, backends=["torchmetrics", "torcheval"],
                settings={"torcheval": {"adaptive_k": True}})
    assert r.confirmed and keys(r) == ["adaptive_k"]
    r = compare("precision_at_k", rel, sc, top_k=5, adaptive_k=True)
    assert r.results["pytrec_eval"].skipped and r.agree


def test_graded_relevance_note():
    r = compare("map", REL, SC)
    assert any("graded" in n for n in r.notes)


@pytest.mark.parametrize("metric,kwargs,msg", [
    ("precision_at_k", {}, "needs top_k"),
    ("ndcg", dict(average="macro"), "takes top_k"),
    ("ndcg", dict(adaptive_k=True), "precision_at_k only"),
    ("f1", dict(empty_queries="skip"), "ranking metrics only"),
    ("ndcg", dict(empty_queries="drop"), "empty_queries must be"),
])
def test_invalid_ranking_settings(metric, kwargs, msg):
    y, p = (REL, SC) if metric != "f1" else ([0, 1], [0, 1])
    with pytest.raises(ValueError, match=msg):
        compare(metric, y, p, **kwargs)


@pytest.mark.parametrize("y,p,msg", [
    ([[1, 0]], [[0.5, 0.2, 0.1]], "2 relevance values but 3 scores"),
    ([[1, -1]], [[0.5, 0.2]], "non-negative integers"),
    ([[1, 0], [1, 0]], [[0.5, 0.2]], "2 queries"),
    (np.array([1, 0]), np.array([0.5, 0.2]), "must be 2-D"),
])
def test_invalid_ranking_inputs(y, p, msg):
    with pytest.raises(ValueError, match=msg):
        compare("ndcg", y, p)


def test_aliases():
    assert compare("NDCG@k", REL, SC).metric == "ndcg"
    assert compare("p@k", REL, SC, top_k=2).metric == "precision_at_k"


def test_batched_not_supported():
    with pytest.raises(ValueError, match="does not support ranking"):
        compare_batched("ndcg", [(REL, SC)])


def test_testing_helper():
    assert_metrics_agree("ndcg", REL, SC, top_k=3)
    assert_metrics_agree("map", REL, TIES, allow=["ranking_ties"])


def test_cli_ranking_files(tmp_path, capsys):
    from metric_diff.cli import main

    (tmp_path / "rel.csv").write_text("3,2,0,1,0\n0,1,0,0,1\n2,0,1\n", encoding="utf-8")
    (tmp_path / "sc.csv").write_text("0.9,0.8,0.7,0.2,0.1\n0.5,0.4,0.9,0.3,0.2\n0.3,0.6,0.1\n", encoding="utf-8")
    code = main(["map", "--true", str(tmp_path / "rel.csv"), "--pred", str(tmp_path / "sc.csv"),
                 "--top-k", "2"])
    out = capsys.readouterr().out
    assert code == 1 and "[map_cutoff]" in out and "3-5 candidates" in out
