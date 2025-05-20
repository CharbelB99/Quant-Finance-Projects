import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, freq=252) -> float:
    return returns.mean() * np.sqrt(freq) / returns.std()

def sortino(returns: pd.Series, target=0.0, freq=252) -> float:
    downside = np.minimum(returns - target/freq, 0.0)
    return returns.mean() * freq / np.sqrt((downside**2).mean() * freq)

def cvar(returns: pd.Series, alpha=0.05) -> float:
    cutoff = returns.quantile(alpha)
    return returns[returns <= cutoff].mean()

def turnover(weights: pd.DataFrame) -> float:
    return weights.diff().abs().sum(axis=1).mean()