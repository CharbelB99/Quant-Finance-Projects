"""
Regime Backtester Package
Version: 0.1.0
"""

__all__ = [
    "DataHandler",
    "FeatureEngineer",
    "build_detector", "KMeansDetector", "HMMDetector",
    "RegimeSignalGenerator",
    "build_allocator", "RiskParityAllocator", "StrengthAllocator",
    "Backtester",
    "sharpe", "sortino", "cvar", "turnover",
]

from .data import DataHandler
from .features import FeatureEngineer
from .detectors import build_detector, KMeansDetector, HMMDetector
from .signals import RegimeSignalGenerator
from .portfolio import build_allocator, RiskParityAllocator, StrengthAllocator
from .backtest import Backtester
from .metrics import sharpe, sortino, cvar, turnover