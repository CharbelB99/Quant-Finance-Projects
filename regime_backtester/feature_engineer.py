import pandas as pd
from typing import Tuple

class FeatureEngineer:
    """Compute rolling features and custom indicators."""
    def __init__(self, price: pd.Series):
        self.price = price.sort_index()
        self.returns = price.pct_change().dropna()

    def compute_basic(self, window: int) -> pd.DataFrame:
        """Volatility, autocorrelation, and rolling Sharpe."""
        vol  = self.returns.rolling(window).std().rename("vol")
        ac   = self.returns.rolling(window).apply(lambda x: x.autocorr(), raw=False).rename("ac")
        mean = self.returns.rolling(window).mean()
        sharpe = (mean / vol).rename("sharpe")
        feats = pd.concat([vol, ac, sharpe], axis=1).dropna()
        return feats

    def compute_rsi(self, window: int) -> pd.Series:
        """Relative Strength Index."""
        delta = self.price.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up   = up.ewm(alpha=1/window, adjust=False).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / roll_down
        return (100 - (100 / (1 + rs))).rename(f"rsi_{window}")

    def compute_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD line and signal line."""
        ema_fast = self.price.ewm(span=fast, adjust=False).mean()
        ema_slow = self.price.ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).rename("macd")
        macd_sig  = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_sig")
        return pd.concat([macd_line, macd_sig], axis=1)
