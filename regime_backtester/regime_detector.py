import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple

class RegimeDetector:
    """Clusters feature space into discrete regimes."""
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.fitted = False

    def fit(self, features: pd.DataFrame) -> pd.Series:
        labels = self.model.fit_predict(features)
        self.fitted = True
        return pd.Series(labels, index=features.index, name="regime")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")
        labels = self.model.predict(features)
        return pd.Series(labels, index=features.index, name="regime")

    def plot_regimes(self, price: pd.Series, regimes: pd.Series, ax=None) -> None:
        """Overlay regime coloration on price chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        for reg in sorted(regimes.unique()):
            mask = regimes == reg
            ax.plot(price[mask].index, price[mask], marker='.', linestyle='None', label=f"R{reg}")
        ax.legend(title="Regime")
        ax.set_title("Price by Regime")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
