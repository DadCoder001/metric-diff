"""metric-diff: find out why ML metric libraries disagree, and how to fix it."""

from .core import Cause, Report, Result, compare
from .inputs import METRICS

__all__ = ["compare", "Report", "Cause", "Result", "METRICS"]
__version__ = "0.1.0.dev0"
