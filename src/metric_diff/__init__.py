"""metric-diff: find out why ML metric libraries disagree, and how to fix it."""

__version__ = "0.1.0.dev0"

from .batched import BatchReport, Finding, compare_batched  # noqa: E402
from .core import Cause, Report, Result, compare  # noqa: E402
from .inputs import METRICS  # noqa: E402
from .matrix import Drift, MatrixResult, run_matrix  # noqa: E402

__all__ = ["compare", "compare_batched", "run_matrix", "Report", "BatchReport", "MatrixResult",
           "Drift", "Cause", "Finding", "Result", "METRICS"]
