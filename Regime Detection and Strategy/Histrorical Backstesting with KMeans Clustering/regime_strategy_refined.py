import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# === Parameters ===
TICKER = "^GSPC"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
MA_WINDOW = 10   # moving average period for signal generation

# Parameter grids for optimization:
POSSIBLE_WINDOWS = [20, 30, 50, 100]         # rolling window sizes
POSSIBLE_CLUSTERS = [2, 3, 4, 5, 6]            # number of clusters for KMeans

# ======================================
# Step 1: Additional Technical Indicators
# ======================================

def compute_rsi(data, period=14):
    """
    Compute Relative Strength Index (RSI) for the 'Close' prices.
    
    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        period (int): Window length for calculating RSI.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Compute MACD (Moving Average Convergence Divergence) indicators.
    
    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        fast_period (int): Span for the fast EMA.
        slow_period (int): Span for the slow EMA.
        signal_period (int): Span for the signal line EMA.
    
    Returns:
        tuple: (macd_line, signal_line, macd_hist)
    """
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_momentum(data, period=10):
    """
    Compute momentum as the percentage change over a specified period.
    
    Parameters:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        period (int): Look-back period for momentum calculation.
    
    Returns:
        pd.Series: Momentum values.
    """
    momentum = data['Close'] / data['Close'].shift(period) - 1
    return momentum

# === 1. Download data ===
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    close = data["Close"]
    close.name = "Close"  # ensure a consistent Series name
    return close

# === 2. Compute features ===
def compute_features(price_series, window):
    """
    Compute a set of features including:
      - Original features: rolling volatility, autocorrelation, local Sharpe ratio.
      - Additional technical indicators: RSI, MACD (line, signal, histogram), and momentum.
    
    Parameters:
        price_series (pd.Series or pd.DataFrame): Series or single-column DataFrame of close prices.
        window (int): Window period for rolling calculations.
    
    Returns:
        tuple: (combined features DataFrame, returns Series)
    """
    # Ensure price_series is a Series. If it's a DataFrame with one column, extract that column.
    if isinstance(price_series, pd.DataFrame):
        if price_series.shape[1] == 1:
            price_series = price_series.iloc[:, 0]
        else:
            raise ValueError("price_series DataFrame must have exactly one column.")
    elif not isinstance(price_series, pd.Series):
        price_series = pd.Series(price_series)
    
    # Compute returns from the price series
    returns = price_series.pct_change().dropna().squeeze()
    returns.name = "Returns"
    
    # Original features from returns
    volatility = returns.rolling(window).std()
    autocorr = returns.rolling(window).apply(lambda x: x.autocorr(), raw=False)
    local_sharpe = returns.rolling(window).mean() / volatility
    features = pd.concat([volatility, autocorr, local_sharpe], axis=1)
    features.columns = ['volatility', 'autocorr', 'sharpe']
    
    # Convert price_series to a DataFrame with preserved index via to_frame()
    price_df = price_series.to_frame(name='Close')
    # Ensure the "Close" column is numeric
    price_df['Close'] = pd.to_numeric(price_df['Close'], errors='coerce')
    
    # Additional technical indicators:
    rsi = compute_rsi(price_df)
    macd_line, signal_line, macd_hist = compute_macd(price_df)
    momentum = compute_momentum(price_df)
    
    technicals = pd.concat([rsi, macd_line, signal_line, macd_hist, momentum], axis=1)
    technicals.columns = ['RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'Momentum']
    
    # Merge original features with technical indicators and drop missing values
    combined = pd.concat([features, technicals], axis=1)
    return combined.dropna(), returns

# === 3. Cluster features into regimes ===
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
        # Example rule based on regime and moving average:
        if regime == 0 and price > moving_avg:
            signal.at[date] = 1  # bullish signal for trend-following
        elif regime == 2 and price < moving_avg:
            signal.at[date] = 1  # mean-reversion signal
    return signal

# === 5. Backtest strategy ===
def backtest_strategy(returns, signals):
    aligned_returns = returns.loc[signals.index]
    strategy_returns = aligned_returns.mul(signals, axis=0)
    cumulative = (1 + strategy_returns).cumprod()

    # Debug prints
    print("\nSignal summary:")
    print(signals.value_counts())
    print("\nStrategy returns head:")
    print(strategy_returns.dropna().head())
    
    return strategy_returns, cumulative

# === 6. Plot performance for one pair ===
def plot_performance(price_series, strategy_curve):
    aligned_price = price_series.loc[strategy_curve.index]
    buy_hold_returns = aligned_price.pct_change().dropna()
    buy_hold_curve = (1 + buy_hold_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold_curve.index, buy_hold_curve, label="Buy & Hold", color='black', linewidth=2.5)
    plt.plot(strategy_curve.index, strategy_curve, label="Regime-Aware Strategy", color='orange', linewidth=2)
    plt.title("Equity Curve: Regime-Aware Strategy vs. Buy & Hold")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 7. Print performance metrics ===
def print_metrics(strategy_returns, buy_hold_returns):
    def metrics(returns):
        cumulative_return = (1 + returns).prod() - 1
        ann_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        ann_return = float(ann_return)
        ann_vol = float(ann_vol)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
        max_dd = ((1 + returns).cumprod().cummax() - (1 + returns).cumprod()).max()
        nonzero = returns[returns != 0]
        win_rate = (nonzero > 0).mean() if len(nonzero) > 0 else np.nan
        return ann_return, ann_vol, sharpe, max_dd, win_rate

    strat = metrics(strategy_returns.dropna())
    hold = metrics(buy_hold_returns.dropna())

    print("\n📊 Strategy Performance Metrics:")
    print(f"{'Metric':<20}{'Strategy':>12}{'Buy & Hold':>15}")
    labels = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    for i in range(len(labels)):
        print(f"{labels[i]:<20}{strat[i]:12.2%}{hold[i]:15.2%}")

# === 8. Optimization: Grid Search for Optimal Window and Cluster Number ===
def optimize_parameters(price_series, possible_windows, possible_clusters, ma_window):
    results = []
    all_curves = {}
    bh_returns = price_series.pct_change().dropna()
    buy_hold_curve = (1 + bh_returns).cumprod()

    for window in possible_windows:
        for clusters in possible_clusters:
            try:
                features, returns = compute_features(price_series, window)
                features, _ = cluster_features(features, clusters)
                signals = build_signals(price_series, features, ma_window)
                strategy_returns, cumulative = backtest_strategy(returns, signals)
                # Align buy & hold returns to strategy period
                bh_returns_aligned = price_series.pct_change().loc[strategy_returns.index]
                if isinstance(bh_returns_aligned, pd.DataFrame):
                    bh_returns_aligned = bh_returns_aligned.squeeze()
                bh_returns_aligned = pd.Series(bh_returns_aligned, index=strategy_returns.index)
                bh_returns_aligned.name = "BuyHold"
                
                cumulative_return = (1 + strategy_returns).prod() - 1
                ann_return = (1 + cumulative_return) ** (252 / len(strategy_returns)) - 1
                ann_vol = strategy_returns.std() * np.sqrt(252)
                ann_return = float(ann_return)
                ann_vol = float(ann_vol)
                sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
                max_dd = ((1 + strategy_returns).cumprod().cummax() - (1 + strategy_returns).cumprod()).max()
                nonzero = strategy_returns[strategy_returns != 0]
                win_rate = (nonzero > 0).mean() if len(nonzero) > 0 else np.nan

                result = {
                    'window': window,
                    'clusters': clusters,
                    'annual_return': ann_return,
                    'volatility': ann_vol,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'win_rate': win_rate,
                    'curve': cumulative
                }
                results.append(result)
                all_curves[(window, clusters)] = cumulative
            except Exception as e:
                print(f"Error with window={window}, clusters={clusters}: {e}")
                continue
    return results, buy_hold_curve, all_curves

# === 9. Plot optimization results (all equity curves) ===
def plot_optimization_results(buy_hold_curve, results):
    plt.figure(figsize=(14, 8))
    plt.plot(buy_hold_curve.index, buy_hold_curve, label="Buy & Hold", color='black', linewidth=3, alpha=0.8)
    
    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    for i, res in enumerate(results):
        label = f"W={res['window']}, C={res['clusters']}, Sharpe={res['sharpe']:.2f}"
        plt.plot(res['curve'].index, res['curve'], color=colors[i], alpha=0.7, label=label)
    plt.title("Equity Curves for Different Parameter Combinations")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    # Load data
    price_series = get_data(TICKER, START_DATE, END_DATE)
    
    # First, show the standard strategy (using default parameters) for reference:
    default_window = 30
    default_clusters = 3
    features, returns = compute_features(price_series, default_window)
    features, _ = cluster_features(features, default_clusters)
    signals = build_signals(price_series, features, MA_WINDOW)
    strategy_returns, cumulative = backtest_strategy(returns, signals)
    
    # Align buy & hold returns with strategy timeframe
    buy_hold_returns = price_series.pct_change().loc[strategy_returns.index]
    if isinstance(buy_hold_returns, pd.DataFrame):
        buy_hold_returns = buy_hold_returns.squeeze()
    buy_hold_returns = pd.Series(buy_hold_returns, index=strategy_returns.index)
    buy_hold_returns.name = "BuyHold"
    
    # Plot the standard strategy vs. buy & hold
    plot_performance(price_series, cumulative)
    print_metrics(strategy_returns, buy_hold_returns)
    
    # === Now optimize parameters ===
    print("\nOptimizing parameters (window size and number of clusters)...")
    results, buy_hold_curve, all_curves = optimize_parameters(price_series, POSSIBLE_WINDOWS, POSSIBLE_CLUSTERS, MA_WINDOW)
    
    summary = pd.DataFrame(results)
    summary = summary.sort_values(by='sharpe', ascending=False)
    print("\nOptimization Summary (sorted by Sharpe Ratio):")
    print(summary[['window', 'clusters', 'annual_return', 'volatility', 'sharpe', 'max_dd', 'win_rate']])
    
    best = summary.iloc[0]
    print("\nOptimal Parameters:")
    print(f"Rolling Window: {best['window']} days, Clusters: {best['clusters']}")
    print(f"Best Sharpe Ratio: {best['sharpe']:.2f}")
    
    plot_optimization_results(buy_hold_curve, results)
