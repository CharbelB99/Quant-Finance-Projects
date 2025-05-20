import pandas as pd

class Allocator:
    def allocate(self, signals: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class RiskParityAllocator(Allocator):
    def __init__(self, vol_lookback=63):
        self.vol_lookback = vol_lookback

    def allocate(self, signals: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        vol = returns.rolling(self.vol_lookback).std()
        inv_vol = 1 / vol
        w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        return w.multiply(signals)

class StrengthAllocator(Allocator):
    def allocate(self, signals: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
        raw = 1 / (distances + 1e-6)
        w = raw.div(raw.sum(axis=1), axis=0)
        return w.multiply(signals)

def build_allocator(cfg: dict) -> Allocator:
    typ = cfg['type'].lower()
    params = cfg.get('params', {})
    if typ == 'risk_parity': return RiskParityAllocator(**params)
    if typ == 'strength': return StrengthAllocator(**params)
    raise ValueError(f"Unknown allocator type: {typ}")