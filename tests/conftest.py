# tests/conftest.py

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def price() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    data = np.linspace(100, 200, len(idx))
    return pd.Series(data, index=idx, name="price")

@pytest.fixture
def signals(price: pd.Series) -> pd.Series:
    cycle = [1, 0, -1]
    sigs = [cycle[i % len(cycle)] for i in range(len(price))]
    return pd.Series(sigs, index=price.index, name="signals")

# Alias fixtures to match your test signatures:
@pytest.fixture(name="price_series")
def price_series_alias(price):
    return price

@pytest.fixture(name="random_signals")
def random_signals_alias(signals):
    return signals
