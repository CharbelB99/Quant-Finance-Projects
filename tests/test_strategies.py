from regime_backtester.strategies import TrendMeanStrategy

def test_trend_mean_signals(price_series):
    # constant regime → all zeros or ±1
    regs = price_series.copy().apply(lambda _: 1)
    strat = TrendMeanStrategy(price_series, regs, ma_window=10)
    sig = strat.generate_signals()
    assert set(sig.unique()) <= {-1, 0, 1}
