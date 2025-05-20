# tests/test_market_data.py
import pytest
from regime_backtester.market_data import MarketData

def test_download_invalid_ticker_raises():
    md = MarketData("XXXXINVALID", "2020-01-01", "2020-01-10", max_retries=1)
    with pytest.raises(Exception):
        md.download()
