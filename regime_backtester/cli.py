import argparse
import yaml
from .config import load_config
from .data import DataHandler
from .features import FeatureEngineer
from .detectors import build_detector
from .signals import RegimeSignalGenerator
from .portfolio import build_allocator
from .backtest import Backtester
from .metrics import sharpe, sortino, cvar, turnover

def main():
    parser = argparse.ArgumentParser(description="Run regime backtest")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data = DataHandler(cfg['symbols'], cfg['start_date'], cfg['end_date'])
    fe   = FeatureEngineer(**cfg['features'])
    det  = build_detector(cfg['detector'])
    sig  = RegimeSignalGenerator(**cfg['signal'])
    alloc= build_allocator(cfg['allocator'])
    bt   = Backtester(data, fe, det, sig, alloc, cfg['costs'])

    pnl = bt.run(walk_forward=True, train_years=cfg['train_years'], test_years=cfg['test_years'])
    print("CAGR/Sharpe/Sortino/CVaR:", sharpe(pnl), sortino(pnl), cvar(pnl))

if __name__ == "__main__":
    main()
