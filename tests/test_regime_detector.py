import pandas as pd
from regime_backtester.regime_detector import RegimeDetector

def test_fit_predict_consistency(price_series):
    # dummy feature: two clusters
    feats = pd.DataFrame({"x": list(range(50)) + list(range(50))},
                         index=price_series.index[:100])
    det = RegimeDetector(n_clusters=2)
    labels = det.fit(feats)
    assert set(labels.unique()) <= {0, 1}
    labels2 = det.predict(feats)
    assert labels2.shape == labels.shape
