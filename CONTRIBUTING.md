# Contributing to metric-diff

Thanks for helping! The most valuable contributions are **new mismatches**: a metric that gives different numbers in different libraries, with the reason. Each one becomes a detector (so `compare()` can explain it) and an edge case (so the conformance matrix watches it across library versions).

## Ways to help

- **Report an unexplained mismatch**: when metric-diff says "no known cause", open an issue with the *Unexplained mismatch* template and the `--json` output.
- **Report a known difference** we don't cover yet: *New edge case* template.
- **Add a detector or an edge case**: see below.
- **Improve the user guide**: scenarios people actually hit are the best documentation.

Questions and ideas are welcome in [Discussions](https://github.com/DadCoder001/metric-diff/discussions).

## Development setup

```bash
git clone https://github.com/DadCoder001/metric-diff
cd metric-diff
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch is enough
pip install -e ".[dev]"
pytest
```

Tests that need a library you haven't installed skip themselves; the Hugging Face `evaluate` tests download metric scripts the first time.

## How metric-diff is organised

| File | What it does |
|---|---|
| `src/metric_diff/inputs.py` | Validates inputs and infers the task (binary, multiclass, regression, segmentation) |
| `src/metric_diff/backends.py` | One adapter per library, calling it the way a user would |
| `src/metric_diff/segmentation.py` | Segmentation adapters (Dice, IoU) |
| `src/metric_diff/text.py` | Text-generation adapters (BLEU, chrF, ROUGE) and their detectors (`TEXT_DETECTORS`) |
| `src/metric_diff/ranking.py` | Ranking/retrieval adapters (NDCG, MAP, MRR, @k) and their detectors (`RANKING_DETECTORS`) |
| `src/metric_diff/detectors.py` | Known causes. Each detector can **align** one setting across libraries |
| `src/metric_diff/core.py` | `compare()`: runs every library, then tries detector alignments until the libraries agree |
| `src/metric_diff/batched.py` | `compare_batched()`: per-batch vs accumulated vs whole-dataset |
| `src/metric_diff/testing.py` | `assert_metrics_agree()` / `assert_batches_consistent()` for users' test suites |
| `src/metric_diff/cases.py` / `matrix.py` | The edge-case library and the conformance matrix |

## Adding a detector

A detector is only useful if it can **prove** its explanation. In `detectors.py` (or `text.py` / `ranking.py` for those metrics):

1. `relevant(case, metric)`: cheap check that the cause could apply (e.g. "predictions are logits").
2. `transform(y_true, y_pred, case)` and/or `options(case, metric)`: how to align the setting across every library (apply sigmoid first, pass `average="macro"`, ...).
3. `explain(case, metric, backends)`: what differs and the fix, in plain language, naming the libraries and settings involved.
4. `requires` / `affects` (or `affected()` when it depends on the settings): which library must be present, and which libraries' results it explains (used to pick the right cause when several alignments work).
5. Add it to `DETECTORS` (`TEXT_DETECTORS`, `RANKING_DETECTORS`). Order matters only when transforms are combined.

Then add a test in `tests/` that shows **both** that the libraries disagree with their defaults **and** that metric-diff reports your cause as confirmed. If the upstream library later fixes the behaviour, the first assertion fails and tells us to update the case.

## Adding an edge case to the matrix

Append an `EdgeCase` to `CASES` in `src/metric_diff/cases.py`: a short id, a title, the metric, a function returning small deterministic `(y_true, y_pred)`, any settings, and the upstream issue if there is one. Keep data tiny; the matrix runs every case on every library.

## Pull requests

- One topic per pull request; describe what changes and link the issue.
- `pytest` must pass. The user guide's Python examples are run as tests too.
- Be precise and kind about other libraries: most differences are reasonable design choices, not bugs. When something does look like a bug, link or file an upstream issue with a minimal reproduction.

## Response times

metric-diff is maintained in spare time. Issues and pull requests are usually looked at within a week; please be patient if it takes longer.

By contributing you agree that your contributions are licensed under the project's [MIT License](LICENSE) and that you'll follow the [Code of Conduct](CODE_OF_CONDUCT.md).
