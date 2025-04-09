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
MA_WINDOW = 10

# === 1. Download data ===
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    close = data["Close"]
    close.name = "Close"
    return close

# === 2. Compute features ===
def compute_features(price_series, window):
    returns = price_series.pct_change().dropna()
    returns = returns.squeeze()
    returns.name = "Returns"

    volatility = returns.rolling(window).std()
    autocorr = returns.rolling(window).apply(lambda x: x.autocorr(), raw=False)
    sharpe = returns.rolling(window).mean() / volatility

    features = pd.concat([volatility, autocorr, sharpe], axis=1)
    features.columns = ['volatility', 'autocorr', 'sharpe']
    return features.dropna(), returns

# === 3. Cluster into regimes ===
def cluster_features(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans.fit_predict(features)
    features['regime'] = regimes
    return features, kmeans

# === 4. Generate regime-aware signals ===
def build_signals(price_series, features, ma_window):
    aligned_price = price_series.loc[features.index]

    if isinstance(aligned_price, pd.DataFrame):
        aligned_price = aligned_price.squeeze()

    ma = aligned_price.rolling(ma_window).mean()
    signal = pd.Series(0, index=features.index)

    for date in features.index:
        regime = features.at[date, 'regime']

        try:
            price = aligned_price.loc[date]
            moving_avg = ma.loc[date]
        except KeyError:
            continue

        if pd.isna(moving_avg):
            continue

        if regime == 0 and price > moving_avg:
            signal.at[date] = 1
        elif regime == 2 and price < moving_avg:
            signal.at[date] = 1

    return signal

# === 5. Backtest strategy ===
def backtest_strategy(returns, signals):
    aligned_returns = returns.loc[signals.index]
    strategy_returns = aligned_returns.mul(signals, axis=0)
    cumulative = (1 + strategy_returns).cumprod()

    print("\nSignal summary:")
    print(signals.value_counts())

    print("\nStrategy returns head:")
    print(strategy_returns.dropna().head())

    return strategy_returns, cumulative

# === 6. Plot performance ===
def plot_performance(price_series, strategy_curve):
    aligned_price = price_series.loc[strategy_curve.index]
    buy_hold = (1 + aligned_price.pct_change().dropna()).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold, label="Buy & Hold", color='steelblue', alpha=0.6)
    plt.plot(strategy_curve, label="Regime-Aware Strategy", color='orange', linewidth=2)
    plt.title("Strategy vs. Buy & Hold")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 7. Evaluate performance ===
def print_metrics(strategy_returns, buy_hold_returns):
    def metrics(returns):
        cumulative_return = (1 + returns).prod() - 1
        ann_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)

        # Force float to avoid ambiguity
        ann_return = float(ann_return)
        ann_vol = float(ann_vol)

        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
        max_dd = ((1 + returns).cumprod().cummax() - (1 + returns).cumprod()).max()
        nonzero = returns[returns != 0]
        win_rate = (nonzero > 0).mean()

        return ann_return, ann_vol, sharpe, max_dd, win_rate

    strat = metrics(strategy_returns.dropna())
    hold = metrics(buy_hold_returns.dropna())

    print("\n📊 Strategy Performance Metrics:")
    print(f"{'Metric':<20}{'Strategy':>12}{'Buy & Hold':>15}")
    labels = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    for i in range(len(labels)):
        print(f"{labels[i]:<20}{strat[i]:12.2%}{hold[i]:15.2%}")

# === Main ===
if __name__ == "__main__":
    price_series = get_data(TICKER, START_DATE, END_DATE)
    features, returns = compute_features(price_series, WINDOW)
    features, kmeans = cluster_features(features, N_CLUSTERS)
    signals = build_signals(price_series, features, MA_WINDOW)
    strategy_returns, cumulative = backtest_strategy(returns, signals)

    # === Fix buy-and-hold returns ===
    buy_hold_returns = price_series.pct_change().loc[strategy_returns.index]
    if isinstance(buy_hold_returns, pd.DataFrame):
        buy_hold_returns = buy_hold_returns.squeeze()
    buy_hold_returns = pd.Series(buy_hold_returns, index=strategy_returns.index)
    buy_hold_returns.name = "BuyHold"

    plot_performance(price_series, cumulative)
    print_metrics(strategy_returns, buy_hold_returns)
