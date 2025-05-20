import pandas as pd

class Backtester:
    def __init__(
        self,
        data_handler,
        feature_engineer,
        detector,
        signal_generator,
        allocator,
        costs: dict
    ):
        self.data_handler = data_handler
        self.fe = feature_engineer
        self.detector = detector
        self.signal_generator = signal_generator
        self.allocator = allocator
        self.tc = costs.get('tc', 0.0)
        self.slippage = costs.get('slippage', 0.0)

    def run(self, walk_forward=False, train_years=3, test_years=1):
        prices = self.data_handler.fetch()
        feats = self.fe.compute(prices)
        self.detector.fit(feats)
        regimes = self.detector.predict(feats)
        signals = self.signal_generator.generate(prices, regimes)
        signals = self._shift_signals(signals)
        returns = prices.pct_change().fillna(0)
        weights = self.allocator.allocate(signals, returns)
        pnl = (weights.shift(1) * returns).sum(axis=1)
        costs = self._apply_costs(weights, prices)
        net_pnl = pnl - costs.sum(axis=1)
        return net_pnl

    def _apply_costs(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        trades = positions.diff().abs()
        tc_cost = trades * prices * self.tc
        slip_cost = trades * prices * self.slippage
        return tc_cost + slip_cost
