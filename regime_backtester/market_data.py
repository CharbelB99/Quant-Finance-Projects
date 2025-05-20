import logging
import time
from typing import Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class MarketData:
    """Fetches and caches price series for a given ticker."""
    def __init__(self, ticker: str, start: str, end: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._series: Optional[pd.Series] = None

    def download(self) -> pd.Series:
        """Download close prices with retry/backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(self.ticker, start=self.start, end=self.end)
                series = df["Close"].dropna()
                if series.empty:
                    raise ValueError("No data returned")
                self._series = series.sort_index()
                return self._series
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    logger.error("Max retries reached, aborting.")
                    raise
                time.sleep(self.retry_delay * attempt)
        # unreachable
        return pd.Series()
