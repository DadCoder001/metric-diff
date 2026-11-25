"""Every ```python example in the user guide must run, so the guide can't silently rot."""

import re
from pathlib import Path

import pytest

pytest.importorskip("torchmetrics")
pytest.importorskip("torcheval")

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "USER_GUIDE.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), flags=re.S)


def test_guide_has_examples():
    assert len(BLOCKS) >= 8


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_guide_example_runs(code, capsys):
    exec(compile(code, str(GUIDE), "exec"), {"__name__": "guide_example"})


def test_documented_claims_match_behaviour():
    import numpy as np

    from metric_diff import compare_batched

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    scores = np.clip(y * 0.25 + rng.random(1000) * 0.75, 0, 1)
    batches = [(y[i:i + 64], scores[i:i + 64]) for i in range(0, 1000, 64)] + [(y[:24], scores[:24])]
    r = compare_batched("accuracy", batches, expected_n=1000, backends=["sklearn"])
    assert [f.key for f in r.findings] == ["batch_average", "duplicates"]  # as printed in scenario 6
