# regime_detection.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

# === Parameters ===
TICKER = "^GSPC"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
WINDOW = 30
N_CLUSTERS = 3

# === Step 1: Download data ===
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data["Adj Close"]

# === Step 2: Feature engineering ===
def compute_features(price_series, window):
    returns = price_series.pct_change().dropna()
    volatility = returns.rolling(window).std()
    autocorr = returns.rolling(window).apply(lambda x: x.autocorr(), raw=False)
    sharpe = returns.rolling(window).mean() / volatility
    features = pd.concat([volatility, autocorr, sharpe], axis=1)
    features.columns = ['volatility', 'autocorr', 'sharpe']
    return features.dropna(), returns

# === Step 3: Clustering ===
def cluster_features(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans.fit_predict(features)
    features['regime'] = regimes
    return features, kmeans

# === Step 4: Visualization ===
def plot_regimes(price_series, features):
    plt.figure(figsize=(14, 6))
    for regime in features['regime'].unique():
        idx = features.index[features['regime'] == regime]
        plt.plot(idx, price_series.loc[idx], label=f"Regime {regime}")
    plt.legend()
    plt.title("Price Series Colored by Market Regime")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

# === Step 5: Hypothesis Testing ===
def compare_returns(returns, features):
    print("T-test between regime pairs:")
    regimes = features['regime'].unique()
    for i in regimes:
        for j in regimes:
            if i < j:
                r1 = returns[features['regime'] == i]
                r2 = returns[features['regime'] == j]
                t_stat, p_val = ttest_ind(r1, r2, equal_var=False)
                print(f"Regime {i} vs {j}: p = {p_val:.4f}")

# === Run everything ===
if __name__ == "__main__":
    price = get_data(TICKER, START_DATE, END_DATE)
    features, returns = compute_features(price, WINDOW)
    features, kmeans = cluster_features(features, N_CLUSTERS)
    plot_regimes(price, features)
    compare_returns(returns, features)
