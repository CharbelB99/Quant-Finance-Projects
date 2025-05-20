import pandas as pd
from abc import ABC, abstractmethod

class Strategy(ABC):
    """Base strategy; produces integer signals (+1 long, 0 flat, -1 short)."""
    def __init__(self, price: pd.Series, regimes: pd.Series):
        self.price = price.sort_index()
        self.regimes = regimes.reindex(self.price.index).fillna(method='ffill').astype(int)

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        ...

class TrendMeanStrategy(Strategy):
    """Long in bullish regimes above MA, mean‐reversion in bearish."""
    def __init__(self, price, regimes, ma_window: int = 20):
        super().__init__(price, regimes)
        self.ma_window = ma_window

    def generate_signals(self) -> pd.Series:
        ma = self.price.rolling(self.ma_window).mean()
        sig = pd.Series(0, index=self.price.index)
        bullish = (self.regimes == 1) & (self.price > ma)
        bearish = (self.regimes == 2) & (self.price < ma)
        sig[bullish] = 1
        sig[bearish] = -1
        return sig
