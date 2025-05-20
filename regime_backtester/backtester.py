# regime_backtester/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, Optional

class Backtester:
    """Applies trading signals to price series, builds an equity curve, and computes performance metrics."""

    def __init__(self, price: pd.Series, signals: pd.Series) -> None:
        """
        Parameters
        ----------
        price : pd.Series
            Time-indexed price series.
        signals : pd.Series
            Integer signals (+1 long, 0 flat, -1 short), aligned to the same index as `price`.
        """
        self.price: pd.Series = price.sort_index()
        # Daily returns, aligned to price
        self.returns: pd.Series = self.price.pct_change().fillna(0)
        # Signals shifted to avoid look-ahead and aligned to returns
        self.signals: pd.Series = signals.reindex(self.returns.index).fillna(0)

        # These will be populated by run()
        self.strategy_ret: Optional[pd.Series] = None
        self.equity: Optional[pd.Series]       = None

    def run(self) -> pd.Series:
        """
        Execute the backtest.

        - Shift signals by one bar to prevent look-ahead
        - Compute daily strategy returns
        - Build a cumulative equity curve

        Returns
        -------
        pd.Series
            The equity curve starting at 1.0
        """
        sig_lagged = self.signals.shift(1).fillna(0)
        self.strategy_ret = self.returns * sig_lagged
        self.equity       = (1 + self.strategy_ret).cumprod()
        return self.equity

    def metrics(self) -> Dict[str, float]:
        """
        Compute standard performance metrics.

        Must call `run()` before calling this.

        Returns
        -------
        Dict[str, float]
            total_return, annualized_return, annualized_vol, sharpe, max_drawdown
        """
        # GUARD: force the user to run() first
        if self.strategy_ret is None or self.equity is None:
            raise RuntimeError("Backtester.run() must be called before metrics().")

        # Drop any NaNs from the strategy returns
        ret = self.strategy_ret.dropna()

        # Total return over the period
        total_return = float(self.equity.iloc[-1] - 1)
        # Annualized return (252 trading days)
        ann_return   = (1 + total_return) ** (252 / len(ret)) - 1
        ann_vol      = float(ret.std() * np.sqrt(252))
        sharpe       = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
        # Maximum drawdown
        max_dd       = float((self.equity.cummax() - self.equity).max())

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        }
