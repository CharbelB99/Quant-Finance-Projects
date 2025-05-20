import pytest
from regime_backtester.backtester import Backtester

def test_metrics_without_run_raises(price_series, random_signals):
    bt = Backtester(price_series, random_signals)
    with pytest.raises(RuntimeError):
        bt.metrics()

def test_backtester_end_to_end(price_series, random_signals):
    bt = Backtester(price_series, random_signals)
    eq = bt.run()
    m = bt.metrics()
    # equity curve starts at 1.0
    assert eq.iloc[0] == pytest.approx(1.0)
    # all metrics present
    for key in ["total_return", "annualized_return", "sharpe", "max_drawdown"]:
        assert key in m
