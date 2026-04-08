import pandas as pd

# -------------------------------------------------------------------
# Outlier rules
# -------------------------------------------------------------------
def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.lt(q1 - k * iqr) | s.gt(q3 + k * iqr)


def detect_outliers_zscore(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=s.index)
    return ((s - s.mean()) / std).abs().gt(threshold)


def detect_outliers_mad(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    median = s.median()
    mad = (s - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=s.index)
    return (0.6745 * (s - median) / mad).abs().gt(threshold)