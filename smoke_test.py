# smoke_test.py

import pandas as pd

# 1) Data
from regime_backtester.data import DataHandler
dh = DataHandler(["SPY", "TLT"], "2020-01-01", "2020-03-01")
prices = dh.fetch()
assert not prices.empty, "DataHandler.fetch() returned empty"

# 2) Features
from regime_backtester.features import FeatureEngineer
fe = FeatureEngineer(vol_lookback=20, rsi_window=5, macd_fast=6, macd_slow=13, macd_signal=5)
features = fe.compute(prices)
assert features.shape[0] == prices.shape[0], "Feature rows mismatch"
assert features.shape[1] > 0, "No features computed"

# 3) Detectors
from regime_backtester.detectors import build_detector
det_cfg = {"type": "kmeans", "params": {"n_clusters": 2}}
det = build_detector(det_cfg)
det.fit(features)
regimes = det.predict(features)
assert len(regimes) == features.shape[0], "Regime labels length mismatch"

# 4) Signals
from regime_backtester.signals import RegimeSignalGenerator
sig = RegimeSignalGenerator(ma_window=10)
signals = sig.generate(prices, regimes)
assert signals.shape == prices.shape, "Signals shape mismatch"
assert set(pd.unique(signals.values.ravel())) <= {-1, 0, 1}, "Unexpected signal values"

# 5) Allocator
from regime_backtester.portfolio import build_allocator
alloc_cfg = {"type":"risk_parity", "params":{"vol_lookback":20}}
alloc = build_allocator(alloc_cfg)
rets = prices.pct_change().fillna(0)
weights = alloc.allocate(signals, rets)
assert weights.shape == signals.shape, "Weights shape mismatch"

# 6) Metrics
from regime_backtester.metrics import sharpe, sortino, cvar, turnover
pnl = (weights.shift(1) * rets).sum(axis=1)
print(f"Sharpe: {sharpe(pnl):.2f}, Sortino: {sortino(pnl):.2f}, CVaR: {cvar(pnl):.4f}, Turnover: {turnover(weights):.4f}")

# 7) Backtest
from regime_backtester.backtest import Backtester
bt = Backtester(dh, fe, det, sig, alloc, {"tc":0.0, "slippage":0.0})
net_pnl = bt.run(walk_forward=False)
assert net_pnl.shape[0] == prices.shape[0], "Backtest P&L length mismatch"
print("Backtest P&L head:\n", net_pnl.head())

# 8) CLI sanity check (optional)
print("Smoke-test passed for all modules.")
