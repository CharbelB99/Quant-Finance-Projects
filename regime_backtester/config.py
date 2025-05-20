import yaml

DEFAULT_CONFIG = {
    "symbols": ["SPY"],
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "train_years": 3,
    "test_years": 1,
    "detector": {"type": "kmeans", "params": {"n_clusters": 3}},
    "features": {"vol_lookback": 63, "rsi_window": 14, "macd_fast":12, "macd_slow":26, "macd_signal":9},
    "signal": {"ma_window": 20},
    "allocator": {"type": "risk_parity", "params": {"vol_lookback":63}},
    "costs": {"tc": 0.0005, "slippage": 0.0005},
    "metrics": ["sharpe","sortino","cvar","turnover"]
}

def load_config(path="config.yaml"):
    """
    Load a YAML config and merge with DEFAULT_CONFIG.
    """
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}
    # deep merge if needed; here shallow
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg or {})
    return merged
