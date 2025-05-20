import pandas as pd
import numpy as np

class RegimeSignalGenerator:
    def __init__(self, ma_window=20):
        self.ma_window = ma_window

    def generate(self, price_df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        # compute moving average
        ma = price_df.rolling(self.ma_window).mean()
        signals = pd.DataFrame(index=price_df.index, columns=price_df.columns)
        for sym in price_df:
            s = np.where(price_df[sym] > ma[sym], 1, -1)
            signals[sym] = s
        # modulate by regimes: e.g., regime 0 = trend follow, regime 2 = mean revert
        # This is just an example mapping:
        signals = signals.where(pd.Series(regimes, index=price_df.index) == 0, -signals)
        return signals.fillna(0)