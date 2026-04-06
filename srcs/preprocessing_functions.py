import numpy as np
import pandas as pd
from sklearn import mixture

def compute_vif_from_corr(X: pd.DataFrame) -> pd.Series:
    """Compute VIF using the inverse of the correlation matrix.

    Faster than regressing each feature against all others.
    Uses pseudo-inverse for numerical stability.
    """

    if X.shape[1] == 0:
        return pd.Series(dtype=float)

    # Standardize features (improves numerical stability)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0).replace(0.0, np.nan)
    Xs = Xs.fillna(0.0)

    # Correlation matrix
    corr = np.corrcoef(Xs.to_numpy(), rowvar=False)

    # Pseudo-inverse (robust to singular matrices)
    inv_corr = np.linalg.pinv(corr)

    # VIF = diagonal of inverse correlation matrix
    vifs = np.diag(inv_corr)

    return pd.Series(vifs, index=X.columns, name="vif")


def vif_prune(X: pd.DataFrame, vif_max: float = 20.0) -> pd.DataFrame:
    """Iteratively remove features with highest VIF."""
    column_dropped = []
    while X.shape[1]:
        v = compute_vif_from_corr(X)
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


import numpy as np
import pandas as pd

def compute_stat_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(index=[
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
        "mad"
    ], columns=df.columns, dtype=float)

    for col in df.columns:
        x = df[col].to_numpy()

        q1 = np.percentile(x, 25)
        median = np.median(x)
        q3 = np.percentile(x, 75)

        mean = np.mean(x)
        var = np.var(x)
        std = np.std(x)

        mad = np.median(np.abs(x - median))

        summary.loc[:, col] = [
            np.min(x),
            q1,
            median,
            q3,
            np.max(x),
            mean,
            var,
            std,
            std / mean if mean != 0 else np.nan,
            np.max(x) - np.min(x),
            mad
        ]

    return summary