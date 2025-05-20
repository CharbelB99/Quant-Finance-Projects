import argparse
from market_data import MarketData
from feature_engineer import FeatureEngineer
from regime_detector import RegimeDetector
from strategies import TrendMeanStrategy
from backtester import Backtester

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start",   default="2015-01-01")
    p.add_argument("--end",     default="2023-12-31")
    p.add_argument("--n_clusters", type=int, default=3)
    p.add_argument("--ma_window",  type=int, default=20)
    args = p.parse_args()

    md = MarketData(args.ticker, args.start, args.end)
    price = md.download()

    fe = FeatureEngineer(price)
    feats = fe.compute_basic(window=30)

    det = RegimeDetector(n_clusters=args.n_clusters)
    regimes = det.fit(feats)

    strat = TrendMeanStrategy(price.reindex(feats.index), regimes, ma_window=args.ma_window)
    signals = strat.generate_signals()

    bt = Backtester(price, signals)
    equity = bt.run()
    print(bt.metrics())

if __name__ == "__main__":
    main()
