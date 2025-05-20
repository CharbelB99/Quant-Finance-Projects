# tests/test_feature_engineer.py
import pandas as pd
import numpy as np
from regime_backtester.feature_engineer import FeatureEngineer

def test_compute_basic_length():
    price = pd.Series(np.linspace(100, 110, 50),
                      index=pd.date_range("2020-01-01", periods=50))
    fe = FeatureEngineer(price)
    feats = fe.compute_basic(window=5)
    assert "vol" in feats and "ac" in feats and "sharpe" in feats
    assert len(feats) == 50 - 5 + 1
