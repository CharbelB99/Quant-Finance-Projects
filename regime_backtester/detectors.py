from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

class RegimeDetector(ABC):
    @abstractmethod
    def fit(self, features: pd.DataFrame): pass
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray: pass

class KMeansDetector(RegimeDetector):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, features: pd.DataFrame):
        self.model.fit(features.dropna())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features.fillna(method='ffill').dropna())

class HMMDetector(RegimeDetector):
    def __init__(self, n_states=3, covariance_type='full', n_iter=1000, random_state=42):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                                 n_iter=n_iter, random_state=random_state)

    def fit(self, features: pd.DataFrame):
        self.model.fit(features.dropna())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features.fillna(method='ffill').dropna())

def build_detector(cfg: dict) -> RegimeDetector:
    typ = cfg['type'].lower()
    params = cfg.get('params', {})
    if typ == 'kmeans': return KMeansDetector(**params)
    if typ == 'hmm': return HMMDetector(**params)
    raise ValueError(f"Unknown detector type: {typ}")
