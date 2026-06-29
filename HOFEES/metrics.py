import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from sklearn.feature_selection import mutual_info_classif
# add at top
import dcor
# ---------------------------------------------------------------------------
# Gaussian Copula MI utilities (continuous estimators)
# ---------------------------------------------------------------------------

# Winitzki approximation
def erfinv(x: np.ndarray) -> np.ndarray:  # type: ignore
    """
    Approximate the inverse error function using the Winitzki (2008) closed-form formula.

    This is a computationally efficient approximation of `scipy.special.erfinv`
    that avoids iterative solvers. It provides reasonable accuracy
    (typical relative error < 1e-3 across most of the domain -1 < x < 1),
    making it suitable for tasks such as rank Gaussianization,
    statistical transforms, and simulations where exact precision is
    not critical.

    Parameters
    ----------
    x : np.ndarray
        Input values in the interval (-1, 1). Values close to -1 or 1 may
        have reduced accuracy due to the approximation.

    Returns
    -------
    np.ndarray
        Approximate values of the inverse error function evaluated at `x`.

    Notes
    -----
    - Reference: S. Winitzki, "A handy approximation for the error function and
    its inverse," 2008.
    - For high-precision applications, prefer `scipy.special.erfinv`.
    Code - Verified
    """
    a = 0.147
    x = np.asarray(x)
    sgn = np.sign(x)
    ln = np.log(1 - x**2)
    first = (2/(np.pi*a) + ln/2)
    inside = first**2 - ln/a
    return sgn * np.sqrt(np.sqrt(inside) - first)


def _rank_gaussianize(x: np.ndarray) -> np.ndarray:
    """
    Rank-based inverse normal transform (INT) of a 1D array.

    This applies a normal-scores (a.k.a. probit) transform to the ranks of `x`
    to make the marginal distribution approximately standard normal.

    Parameters
    ----------
    x : numpy.ndarray
        1D numeric array.

    Returns
    -------
    numpy.ndarray
        1D float array of the same length as `x`, containing approximately
        standard normal scores (mean ~0, variance ~1). The transform is
        monotone in `x`; tied values receive identical z-scores.

    Notes
    -----
    - The plotting position u = (r - 0.5)/n is the classical **Rankit** choice.
    Code - Verified
    """

    # Convert to float
    x = np.asarray(x).astype(float)
    n = x.shape[0]
    # Each value gets replaced with its rank position in the dataset (1 = smallest, n = largest).
    # Tied values are given the average rank.
    ranks = pd.Series(x).rank(method="average").to_numpy()
    # Convert ranks → uniform distribution (0,1) while avoiding exact values of 0 and 1
    u = (ranks - 0.5) / n
    # This converts uniform(0,1) values into values distributed like a standard Gaussian (mean 0, variance 1).
    z = np.sqrt(2) * erfinv(2*u - 1)
    return z



def _covariance_matrix(Z: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix for multivariate data.

    Assumes rows are observations (samples) and columns are variables (features).
    Each column is mean-centered before computing covariance:

        Cov = (Z_centered.T @ Z_centered) / (n_samples - 1)

    Parameters
    ----------
    Z : np.ndarray of shape (n_samples, n_features)
        Input data where each row is a sample and each column is a variable.

    Returns
    -------
    cov : np.ndarray of shape (n_features, n_features)
        Covariance matrix. Entry [i, j] is the covariance between variables
        i and j. Diagonal entries are variances of the individual variables.

    Notes
    -----
    - Equivalent to `np.cov(Z, rowvar=False)`.
    - Uses (n_samples - 1) in the denominator for an unbiased estimate.
    Code - Verified
    """

    # Convert to NumPy
    Z = np.asarray(Z)
    # Mean-center each column
    Z = Z - Z.mean(axis=0, keepdims=True)
    # Compute covariance
    cov = np.dot(Z.T, Z) / (Z.shape[0] - 1)
    return cov



def _gaussian_mi_multivariate(Zx: np.ndarray, Zy: np.ndarray) -> float:
    """
    Estimate the mutual information (MI) between two random variables
    under the assumption of a joint Gaussian distribution.

    The closed-form formula for Gaussian MI is:
        I(X; Y) = 0.5 * log( |Σ_X| * |Σ_Y| / |Σ| )

    where:
        - Σ_X is the covariance matrix of X,
        - Σ_Y is the covariance matrix of Y,
        - Σ   is the joint covariance matrix of [X, Y].

    Parameters
    ----------
    Zx : np.ndarray, shape (n_samples, d_x)
        Samples of the first random variable (X).
    Zy : np.ndarray, shape (n_samples, d_y)
        Samples of the second random variable (Y).

    Returns
    -------
    float
        Estimated mutual information in nats (natural log base).

    Notes
    -----
    - This estimator assumes X and Y follow a joint Gaussian distribution.
    - If covariance matrices are not positive definite due to numerical issues
      (e.g., near-collinearity or rounding), a small ridge (1e-8) is added
      to the diagonal to stabilize log-determinant calculations.
    - The result is always ≥ 0, with higher values indicating stronger
      statistical dependence between X and Y.
    Code - Verified
    """
    # Covariance of X
    Σx = _covariance_matrix(Zx)
    # Covariance of Y
    Σy = _covariance_matrix(Zy)
    # Joint covariance of [X, Y]
    Zxy = np.column_stack([Zx, Zy])
    Σ = _covariance_matrix(Zxy)

    # Compute log-determinants (numerically stable)
    signx, logdetx = slogdet(Σx)
    signy, logdety = slogdet(Σy)
    sign, logdet = slogdet(Σ)

    # If any determinant is invalid (≤ 0), add ridge regularization
    if signx <= 0 or signy <= 0 or sign <= 0:
        ridge = 1e-8
        Σx = Σx + np.eye(Σx.shape[0]) * ridge
        Σy = Σy + np.eye(Σy.shape[0]) * ridge
        Σ = Σ + np.eye(Σ.shape[0]) * ridge
        signx, logdetx = slogdet(Σx)
        signy, logdety = slogdet(Σy)
        sign, logdet = slogdet(Σ)

    # Return Gaussian MI formula
    return 0.5 * (logdetx + logdety - logdet)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:

    """
    Compute distance correlation using the dcor library.

    Returns a value in [0, 1]:
    - 0 → independence
    - 1 → perfect dependence

    Automatically handles constant inputs by returning 0.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    # dcor returns 0 when either input is constant; cast to float for safety
    return float(dcor.distance_correlation(x, y))


def score_with_target(
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    is_categorical: bool = False,
) -> float:
    """
    Compute a dependency score between a single feature and the target.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of feature values.
    y : np.ndarray
        One-dimensional array of target values.
    task : str
        Type of learning task. 
        - "regression": Uses distance correlation.
        - "classification": Uses `mutual_info_classif`.
    is_categorical : bool, default=False
        Whether `x` should be treated as discrete for classification.
    Returns
    -------
    float
        Estimated dependency score between feature x and target y.
    Code - Verified
    """


    # Coerce inputs to 1-D NumPy arrays
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if task == "regression":
        # Use distance correlation instead of Gaussian mutual information
        return _distance_correlation(x, y)
    else:
        # mutual_info_classif expects X as 2D: (n_samples, n_features)
        X_2d = x.reshape(-1, 1)

        mi = mutual_info_classif(
            X_2d,
            y,
            discrete_features=bool(is_categorical)
        )
        # mi is an array of length 1 (one feature)
        return float(mi[0])

def _gauss_mi_with_target(x: np.ndarray, y: np.ndarray) -> float:
    """
    Gaussian MI between a single 1D feature x and target y.
    Uses rank Gaussianization on each, then the closed-form Gaussian MI.
    Code - Verified
    """
    # Coerce inputs to 1-D NumPy arrays
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Zx = _rank_gaussianize(x).reshape(-1, 1)
    Zy = _rank_gaussianize(y).reshape(-1, 1)
    return _gaussian_mi_multivariate(Zx, Zy)
