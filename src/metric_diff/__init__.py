"""metric-diff: find out why ML metric libraries disagree, and how to fix it."""

from .batched import BatchReport, Finding, compare_batched
from .core import Cause, Report, Result, compare
from .inputs import METRICS

__all__ = ["compare", "compare_batched", "Report", "BatchReport", "Cause", "Finding", "Result", "METRICS"]
__version__ = "0.1.0.dev0"
