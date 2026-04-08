import numpy as np
import pandas as pd
from sklearn import mixture
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X: pd.DataFrame | np.ndarray) -> pd.Series:
    """
    Compute VIF values exactly as statsmodels'
    variance_inflation_factor(exog, i) applied column-by-column.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Design matrix. No constant is added automatically, matching statsmodels.

    Returns
    -------
    pd.Series
        VIF values indexed by column name when X is a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        columns = X.columns
        exog = X.to_numpy(dtype=float, copy=False)
    else:
        exog = np.asarray(X, dtype=float)
        if exog.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame.")
        columns = pd.RangeIndex(exog.shape[1])

    if exog.ndim != 2:
        raise ValueError("X must be a 2D array or DataFrame.")

    if exog.shape[1] == 0:
        return pd.Series(dtype=float, name="vif")

    if not np.isfinite(exog).all():
        raise ValueError("VIF requires finite numeric values. Clean NaN/±inf first.")

    vif_values = np.fromiter(
        (variance_inflation_factor(exog, i) for i in range(exog.shape[1])),
        dtype=float,
        count=exog.shape[1],
    )

    return pd.Series(vif_values, index=columns, name="vif")


def vif_prune(X: pd.DataFrame, vif_max: float = 20.0) -> pd.DataFrame:
    """Iteratively remove features with highest VIF."""
    column_dropped = []
    while X.shape[1]:
        v = compute_vif(X)
        if v.max() <= vif_max:
            break
        X = X.drop(columns=[v.idxmax()])
        column_dropped.append(v.idxmax())
    return X, column_dropped


def bic_grid(x: np.ndarray, max_k: int, iterations: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo BIC grid:
    - x: (n, 1)
    - returns (bic_ite, seeds) where bic_ite shape is (iterations, max_k-1)
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**16, size=(iterations, max_k - 1))
    bic_ite = np.empty((iterations, max_k - 1), dtype=float)

    for i in range(iterations):
        for k in range(1, max_k):
            gmm = mixture.GaussianMixture(n_components=k, random_state=int(seeds[i, k - 1]))
            gmm.fit(x)
            bic_ite[i, k - 1] = gmm.bic(x)
    return bic_ite, seeds

def best_gmm_by_bic(x: np.ndarray, k: int, seeds: np.ndarray) -> mixture.GaussianMixture:
    """Fit k-component GMM over multiple seeds and return the lowest-BIC model."""
    best, best_bic = None, np.inf
    for s in seeds:
        gmm = mixture.GaussianMixture(n_components=k, random_state=int(s))
        gmm.fit(x)
        b = gmm.bic(x)
        if b < best_bic:
            best, best_bic = gmm, b
    assert best is not None
    return best

def compute_stat_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a statistical summary for each column in a DataFrame.

    Rows:
        min, q1, median, q3, max, mean, variance, std_dev,
        coef_variation, range, mad, skewness, kurtosis

    Notes:
    - NaN values are ignored.
    - Variance and standard deviation use population formulas (ddof=0).
    - Kurtosis is excess kurtosis:
        0 for a normal distribution,
        >0 for heavier tails,
        <0 for lighter tails.
    """
    summary = pd.DataFrame(
        index=[
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "variance",
            "std_dev",
            "coef_variation",
            "range",
            "mad",
            "skewness",
            "kurtosis",
        ],
        columns=df.columns,
        dtype=float,
    )

    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

        if x.size == 0:
            summary.loc[:, col] = np.nan
            continue

        q1 = np.percentile(x, 25)
        median = np.median(x)
        q3 = np.percentile(x, 75)

        mean = np.mean(x)
        var = np.var(x, ddof=0)
        std = np.std(x, ddof=0)
        mad = np.median(np.abs(x - median))
        x_range = np.max(x) - np.min(x)

        # Standardized moments
        if std == 0:
            skewness = 0.0
            kurtosis = 0.0
        else:
            z = (x - mean) / std
            skewness = np.mean(z**3)
            kurtosis = np.mean(z**4) - 3.0  # excess kurtosis

        coef_variation = std / mean if mean != 0 else np.nan

        summary.loc[:, col] = [
            np.min(x),
            q1,
            median,
            q3,
            np.max(x),
            mean,
            var,
            std,
            coef_variation,
            x_range,
            mad,
            skewness,
            kurtosis,
        ]

    return summary