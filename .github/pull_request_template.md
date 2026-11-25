## What this changes

<!-- One or two sentences. Link the issue: Fixes #123 -->

## Checklist

- [ ] Tests added or updated (a new detector needs a test that shows the libraries disagree *and* that metric-diff confirms the cause)
- [ ] New mismatches also added as an edge case in `src/metric_diff/cases.py`, linked to the upstream issue if there is one
- [ ] `pytest` passes locally
- [ ] README / user guide updated if behaviour changed
