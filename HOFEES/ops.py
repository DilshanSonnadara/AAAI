import numpy as np
import pandas as pd
from typing import Dict, Callable

# ---------------------------------------------------------------------------
# Safe operator library
# ---------------------------------------------------------------------------

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    """
    Apply a numerically safe log(1 + x) transformation.

    Ensures the input is valid for the log1p function by clipping 
    values below -0.999999 (since log(1 + x) is undefined for x <= -1).

    Parameters
    ----------
    x : np.ndarray
        Input array of values.

    Returns
    -------
    np.ndarray
        Transformed array with log(1 + x) applied element-wise.
    Code - Verified
    """
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, a_min=-0.999999, a_max=None))


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    """
    Apply a numerically safe square-root transformation.

    Clips negative values to 0 before applying sqrt, ensuring 
    the result is always real-valued and avoiding NaNs from 
    invalid square-root operations.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.

    Returns
    -------
    np.ndarray
        Transformed array with sqrt applied element-wise to 
        non-negative values (negatives clipped to 0).
    Code - Verified
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.clip(x, a_min=0.0, a_max=None))


def _safe_div(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Safe division that handles exact zeros AND near-zero values to prevent 
    feature explosions.
    - If |b| ≤ epsilon, the result is set to 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # 1. Create a mask for valid denominators (magnitude > epsilon)
    #    We use abs(b) so we catch both positive and negative near-zeros
    mask = np.abs(b) > epsilon
    
    # 2. Use np.divide with 'where' to avoid RuntimeWarnings
    #    'out' initializes the result (defaulting to 0.0 for invalid cases)
    result = np.zeros_like(a)
    np.divide(a, b, out=result, where=mask)
    
    return result


def _safe_reciprocal(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Safe reciprocal (1/x) that handles exact zeros AND near-zero values 
    to prevent feature explosions.
    If |x| ≤ epsilon → output = 0
    """
    x = np.asarray(x, dtype=float)
    
    # 1. Create a mask for valid denominators (magnitude > epsilon)
    #    This protects against both 0.0 and tiny values like 1e-12
    mask = np.abs(x) > epsilon
    
    # 2. Use np.divide with 'where' to avoid RuntimeWarnings
    #    Initialize result with 0.0 (the default for invalid cases)
    result = np.zeros_like(x)
    
    # We divide 1.0 by x, but ONLY where the mask is True
    np.divide(1.0, x, out=result, where=mask)
    
    return result

"""
    A dictionary of safe unary (single-variable) transformations 
    applied feature-wise during AutoFE. 

    Each operator maps a 1D array → transformed 1D array. 
    These help capture non-linearities and alternative scales 
    that might better relate features to the target.

    Operators
    ---------
    - identity : returns the input as float (no transformation).
    - log1p    : applies a numerically safe log(1+x) transformation by clipping values below -0.999999.
    - sqrt     : applies square-root, stabilizing large values.
    - square   : squares the values, capturing quadratic effects.
    - abs      : absolute value, removes sign while preserving magnitude.
    -reciprocal: reciprocal: returns 1/x, outputting 0 when |x| ≤ ε.
    - cube   :   takes the values to the power of 3.

    
    Code - Verified
"""

UNARY_OPS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: np.asarray(x, dtype=float),  
    "log1p": _safe_log1p,                              
    "sqrt": _safe_sqrt,                                
    "square": lambda x: np.asarray(x, dtype=float) ** 2.0,  
    "abs": lambda x: np.abs(np.asarray(x, dtype=float)),        
    "reciprocal": _safe_reciprocal,
    "cube": lambda x: np.asarray(x, dtype=float) ** 3.0,

}


def _safe_norm_ratio(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute a normalized ratio between two arrays.

    The ratio is computed as a / (|a| + |b| + epsilon), where `epsilon`
    prevents division by zero and ensures numerical stability.

    Parameters
    ----------
    a : np.ndarray
        Numerator array.
    b : np.ndarray
        Second input array.
    epsilon : float, default=1e-6
        Small constant added to the denominator.

    Returns
    -------
    np.ndarray
        Element-wise normalized ratio.
    Code - Verified
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    denom = np.abs(a) + np.abs(b) + epsilon
    
    result = a / denom
    
    return result

"""
    A dictionary of safe binary (two-variable) transformations 
    applied feature-pairwise during AutoFE.

    Each operator maps two 1D arrays (a, b) → transformed 1D array.
    These capture potential interactions or relationships between 
    pairs of features.

    Operators
    ---------
    - add       : element-wise addition (a + b).
    - sub       : element-wise subtraction (a - b).
    - mul       : element-wise multiplication (a * b).
    - div       : Division uses _safe_div, returning 0 whenever the denominator magnitude is ≤ ε.
    - abs_diff  : absolute difference |a - b|, ignores sign but preserves distance.
    - min       : element-wise minimum, captures "lower bound" effect.
    - max       : element-wise maximum, captures "upper bound" effect.


    Notes
    -----
    - All operations cast inputs to float for consistency.
    - Division uses `_safe_div` to avoid infinities and NaNs.
    - `norm_ratio` produces values in [-1, 1], useful as a normalized comparison.

    Code - Verified
"""

BINARY_OPS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {

    "add": lambda a, b: np.asarray(a, dtype=float) + np.asarray(b, dtype=float),
    "sub": lambda a, b: np.asarray(a, dtype=float) - np.asarray(b, dtype=float),
    "mul": lambda a, b: np.asarray(a, dtype=float) * np.asarray(b, dtype=float),
    "div": _safe_div,
    "abs_diff": lambda a, b: np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)),
    "min": lambda a, b: np.minimum(np.asarray(a, dtype=float), np.asarray(b, dtype=float)),
    "max": lambda a, b: np.maximum(np.asarray(a, dtype=float), np.asarray(b, dtype=float)),
    "norm_ratio": _safe_norm_ratio,
}

