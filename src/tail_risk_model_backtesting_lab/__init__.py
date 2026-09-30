"""Tail-risk estimation and coverage backtesting tools."""

from .distributions import SCENARIOS
from .models import METHODS

__all__ = ["METHODS", "SCENARIOS"]
__version__ = "0.1.0"
