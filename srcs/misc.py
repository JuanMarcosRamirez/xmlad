import pandas as pd

def stat_summary(s: pd.Series) -> pd.Series:
    q = s.quantile([0.25, 0.50, 0.75])
    return pd.Series(
        {
            "count": s.count(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "25%": q.loc[0.25],
            "50%": q.loc[0.50],
            "75%": q.loc[0.75],
            "max": s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
        }
    )