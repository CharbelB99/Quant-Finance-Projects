import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, vol_lookback=63, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
        self.vol_lookback = vol_lookback
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        # Returns DataFrame: date × features (multi-index: symbol, feature)
        returns = price_df.pct_change()
        # Rolling volatility
        vol = returns.rolling(self.vol_lookback).std()
        # Autocorrelation (1-lag)
        ac = returns.rolling(self.vol_lookback).apply(lambda x: x.autocorr(), raw=False)
        # RSI
        delta = price_df.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_window).mean()
        avg_loss = loss.rolling(self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # MACD
        ema_fast = price_df.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = price_df.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        # Momentum
        mom = price_df.diff(self.vol_lookback)

        # Combine
        features = pd.concat({
            'vol': vol,
            'autocorr': ac,
            'rsi': rsi,
            'macd': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'momentum': mom
        }, axis=1)
        # Flatten columns: (feat, symbol) -> symbol_feat
        features.columns = [f"{feat}_{sym}" for feat, sym in features.columns]
        return features