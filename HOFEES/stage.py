# autofe/stage.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .ops import UNARY_OPS, BINARY_OPS
from .metrics import score_with_target, _gauss_mi_with_target, _distance_correlation

# ============================================================
# One-stage FE block:
#   - Unary pass (either full or identity-only)
#   - Interaction mining for ONE level (2-way on the stage input)
#   - Best-per-pair binary by score
#   - Can require that each pair includes at least one "prev-new" feature
# ============================================================


@dataclass
class AutoFEStage(TransformerMixin, BaseEstimator):
    """
    One feature-engineering (FE) **stage** that:
      1) applies per-column **unary** operators and keeps the best transform per column (by score with target),
      2) mines 2-way interactions by evaluating all unordered pairs (respecting require_prev_new when applicable). 
      3) For each unordered pair, tries all binary operators, keeps at most one, and keeps it only if its score
         with the target is strictly greater than the score of each individual operand (measured on the chosen unary versions).

    Parameters
    ----------
    task : {"classification", "regression"}, default="classification"
        Learning task type. 
    unary_ops : Dict[str, Callable], default=UNARY_OPS
        Mapping name -> unary transform function f(x) -> np.ndarray.
    binary_ops : Dict[str, Callable], default=BINARY_OPS
        Mapping name -> binary transform function g(a, b) -> np.ndarray.
    random_state : int, default=42
        Random seed for reproducability

    Attributes
    ----------
    feature_names_in_ : List[str]
        Original input column names (as seen at `fit`).
    unary_choice_ : Dict[int, Tuple[str, float]]
        For each column index j: (best_unary_name, best_score).
    unary_ops_used_ : List[str]
        Ordered list of the best unary op name chosen per input column (aligned with `feature_names_in_`).
    binary_specs_ : List[Tuple[int,int,str,str]]
        Each element: (i, j, op_name, new_name) describing a kept binary transform for pair (i, j).
    base_colnames_ : List[str]
        Names for the stage's base (unary-transformed) columns used in `transform`.
    new_binary_mask_ : np.ndarray[bool] or None
        Boolean mask indicating which **output columns of this stage** are newly created binaries
        (same length as the stage's output columns; last `len(binary_specs_)` entries are True).
    - `score_eps`: minimum improvement required over identity for unary transforms.
    - `score_binary`: minimum improvement required over both operands for binary features.

    - Categorical features are not subjected to unary search (identity only),
    and are excluded from binary interaction generation.
    - Binary features are evaluated using Gaussian mutual information (regression)
    or mutual information (classification).
    - For each unordered pair, at most one binary feature is retained.
    - All transformations use numerically safe operators to prevent overflow and invalid values.
    """

    task: str = "classification"                                 # task control flag
    unary_ops: Dict[str, Callable] = field(default_factory=lambda: UNARY_OPS)  # unary operator lib
    binary_ops: Dict[str, Callable] = field(default_factory=lambda: BINARY_OPS) # binary operator lib
    random_state: Optional[int] = 42                             # RNG seed

    
    score_eps: float = 0.01          # margin to beat identity for unary
    score_binary: float = 0.1        # margin to beat operands for binaries

    # Fit artifacts
    feature_names_in_: List[str] = field(init=False, default_factory=list)                 # original column names
    unary_choice_: Dict[int, Tuple[str, float]] = field(init=False, default_factory=dict)  # col -> (op_name, score)
    unary_ops_used_: List[str] = field(init=False, default_factory=list)                   # best unary names
    binary_specs_: List[Tuple[int, int, str, str]] = field(init=False, default_factory=list)  # (i,j,op,new_name)
    base_colnames_: List[str] = field(init=False, default_factory=list)                    # names for base layer

    new_binary_mask_: Optional[np.ndarray] = None                 # marks newly created binary columns

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],                       # input matrix
        y: Union[pd.Series, np.ndarray],                          # target vector
        *,
        require_prev_new: bool = False,                           # growth constraint flag for higher stages
        prev_new_mask: Optional[np.ndarray] = None,               # mask marking "previously-new" columns
        unary_mode: str = "full",                                 # {"full": search best unary, "identity": pass-through}
        feature_types: Optional[dict[str, str]] = None,           # optional guard for categorical-derived cols
    ) -> "AutoFEStage":
        """
        Fit the stage:
          - Unary layer: either identity passthrough or best-unary-per-column by distance correlation
            to the target (both tasks).
          - Binary pass: for all pairs, pick best binary op by score with target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        require_prev_new : bool, default=False
            If True, only consider pairs where **at least one operand** is flagged as previously-new
            (enforces growth of interaction order across stages).
        prev_new_mask : np.ndarray[bool] or None, default=None
            Mask aligned with the **current base layer columns** indicating which are previously-new
            (required if `require_prev_new=True`).
        unary_mode : {"full", "identity"}, default="full"
            "full": for each column, search all unary ops and keep the argmax-score transform
                (distance correlation vs. target);
            "identity": keep columns as-is (no search).

        Returns
        -------
        self : AutoFEStage
            Fitted stage with artifacts populated.
        """
        X_df = pd.DataFrame(X)                                     # normalize input to DataFrame
        y_arr = np.asarray(y).reshape(-1)                          # ensure 1-D numpy array for target
        self.feature_names_in_ = list(X_df.columns)                # store original column names
        feature_types = feature_types or {} #feature_types is the mapping you pass into AutoFEStage.fit that says, for each input column, whether it’s "categorical" or "numeric"
        base_types: list[str] = [] #stores the type of each base (unary-transformed) column in order—either "categorical" or "numeric" based on feature_types

        # ---------- Unary layer ----------
        unary_vals = []                                            # will store best-unary-transformed columns
        self.unary_ops_used_.clear()                               # reset chosen unary names
        self.unary_choice_.clear()                                 # reset per-column choices
        self.base_colnames_.clear()                                # reset output base names

        for j, col in enumerate(X_df.columns):                     # loop through each input column
            x = X_df[col].to_numpy()                               # extract as ndarray
            ftype = feature_types.get(col, "numeric")
            is_cat = ftype == "categorical"
            if self.task == "classification":
                def _score_unary(arr: np.ndarray) -> float:
                    return _distance_correlation(arr, y_arr)       # distance corr for classification
            else:
                def _score_unary(arr: np.ndarray) -> float:
                    return score_with_target(arr, y_arr, task=self.task, is_categorical=is_cat)

            if unary_mode == "identity" or is_cat:     # identity passthrough mode
                vals = UNARY_OPS["identity"](x)                    # apply identity transform
                score = _score_unary(vals)                          # score vs target
                best_name = "identity"                             # record op name
            else:
                # Always compute identity as the baseline
                id_vals = UNARY_OPS["identity"](x)
                id_score = _score_unary(id_vals)
                best_name, best_vals, best_score = "identity", id_vals, id_score

                for name, func in self.unary_ops.items():          # iterate over unary operator library
                    if name == "identity":            # already scored above; skip duplicate work
                        continue
                    try:
                        v = func(x)                                # transform
                        m = _score_unary(v)                        # score
                        if m > best_score:                            # keep best-so-far
                            best_name, best_vals, best_score = name, v, m
                    except Exception:                              # be robust to invalid transforms
                        continue
                
                # **Margin check**: only keep non-identity if strictly better than identity + ε
                if best_name != "identity" and not (best_score > id_score + self.score_eps):
                    best_name, best_vals, best_score = "identity", id_vals, id_score
                
                vals, score = best_vals, best_score                      # finalize chosen values/score

            unary_vals.append(vals)                                # append chosen unary values to matrix builder
            self.unary_ops_used_.append(best_name)                 # remember chosen op name
            self.unary_choice_[j] = (best_name, float(score))         # store (op, score) for this column
            self.base_colnames_.append(                            # set stage's base column name for this feature
                f"U[{best_name}]({col})" if unary_mode == "full" else col
            )
            base_types.append(ftype)

        U = np.column_stack(unary_vals) if unary_vals else np.empty((len(X_df), 0))  # build base layer matrix with unary transformed values

        d = U.shape[1]                                             # number of base columns in this stage
        if d <= 1:                                                 # if <2 columns, no pairs to mine
            self.binary_specs_ = []                                 # clear binary specs
            self.new_binary_mask_ = np.zeros(0, dtype=bool)         # no new columns -It creates an empty 1‑D NumPy array of dtype bool with length 0
            return self                                            # done

        # score per base column
        score_u = [self.unary_choice_[j][1] for j in range(d)]

        # Mutual information score per base column (used for binary comparisons in both tasks)
        if self.task == "regression":
            gmi_u = [_gauss_mi_with_target(U[:, j], y_arr) for j in range(d)]
        else:
            gmi_u = [
                score_with_target(
                    U[:, j],
                    y_arr,
                    task=self.task,
                    is_categorical=(base_types[j] == "categorical"),
                )
                for j in range(d)
            ]

        prev_set: Optional[set[int]] = None                        # set of indices considered "previously-new"
        if require_prev_new:                                       # if growth constraint is active
            if prev_new_mask is None or len(prev_new_mask) != d:   # sanity check for provided mask
                raise ValueError("prev_new_mask must be provided and aligned with columns when require_prev_new=True")
            prev_set = {idx for idx, flag in enumerate(prev_new_mask) if flag}  # build index set
            #It iterates over the boolean prev_new_mask and collects the indices where the flag is True into a set.
            #That set (prev_set) marks which base columns are considered “previously new,” used later to allow only pairs that include at least one of those columns.
        # ---------- Binary pass: pick best operator per allowed pair ----------

        # Build ALL unordered allowed pairs (respect require_prev_new growth rule)
        allowed_pairs: list[tuple[int, int]] = []
        if prev_set is not None:
            for i in range(d):
                for j in range(i + 1, d):
                    if (i in prev_set) or (j in prev_set):
                        allowed_pairs.append((i, j))
        else:
            for i in range(d):
                for j in range(i + 1, d):
                    allowed_pairs.append((i, j))   

        # Guard: drop any pair that involves a categorical-derived base column
        if base_types:
            allowed_pairs = [
                (i, j)
                for (i, j) in allowed_pairs
                if base_types[i] != "categorical" and base_types[j] != "categorical"
            ]

        self.binary_specs_.clear()                                  # reset kept binaries

        for (i, j) in allowed_pairs:                                # loop over shortlisted pairs
            xi, xj = U[:, i], U[:, j]                               # retrieve operands
            best_name, best_vals, best_score = None, None, -np.inf     # search best binary op by score
            for name, func in self.binary_ops.items():              # iterate ops
                try:
                    v = func(xi, xj)                                # candidate binary transform
                    # CHANGED: regression uses Gaussian MI; classification keeps existing scoring
                    if self.task == "regression":
                        m = _gauss_mi_with_target(v, y_arr)
                    else:
                        m = score_with_target(v, y_arr, task=self.task)
                    if m > best_score:                                 # keep best-so-far
                        best_name, best_vals, best_score = name, v, m
                except Exception:                                    # robustness to failures
                    continue
            if best_name is not None and np.isfinite(best_score):
                if self.task == "regression":
                    # Must beat the Gaussian MI of each operand (measured on chosen unary versions)
                    ok = (best_score > gmi_u[i] + self.score_binary) and (best_score > gmi_u[j]+ self.score_binary)
                else:
                    # Use mutual information for operand comparisons in classification as well
                    ok = (best_score > gmi_u[i]+ self.score_binary) and (best_score > gmi_u[j]+ self.score_binary)
                if ok:
                    name_i = self.base_colnames_[i]                      # readable operand names
                    name_j = self.base_colnames_[j]
                    new_name = f"B[{best_name}]({name_i},{name_j})"      # deterministic new column name
                    self.binary_specs_.append((i, j, best_name, new_name))  # record kept binary

        # Mark newly created columns for this stage (last block are new binaries)
        self.new_binary_mask_ = np.array([False] * (d + len(self.binary_specs_)), dtype=bool)
        if len(self.binary_specs_) > 0:
            self.new_binary_mask_[-len(self.binary_specs_):] = True  # tail positions are new

        return self                                                 # fitted

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Reconstruct all **stage outputs** (base/unary layer + kept binaries) given new data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data with the **same columns/order** seen at fit time.

        Returns
        -------
        pd.DataFrame
            The stage’s full output matrix with deterministic column names:
            - Base layer columns (unary/identity), named as in `base_colnames_`
            - Followed by all kept binary columns, in recorded order.
        
        Code - Verified
        """
        X_df = pd.DataFrame(X)                                      # normalize input
        if len(X_df.columns) != len(self.feature_names_in_):        # strict shape check
            raise ValueError("Input has different number of columns than seen in fit().")

        # Base layer reconstruction using the **chosen** unary op per column
        base_vals = []                                              # reconstructed base matrix
        for j, col in enumerate(self.feature_names_in_):            # iterate original columns in order
            op_name, _ = self.unary_choice_[j]                      # chosen op name from fit()
            if op_name not in UNARY_OPS:                            # safety
                raise ValueError(f"Unknown unary op '{op_name}' in unary_choice_; expected one of {sorted(UNARY_OPS)}")
            vals = UNARY_OPS[op_name](X_df[col].to_numpy())         # apply unary op to new data
            base_vals.append(vals)
        U = np.column_stack(base_vals) if base_vals else np.empty((len(X_df), 0))  # base matrix
        cols = list(self.base_colnames_)                            # start column names list

        # Binary reconstruction (apply kept binary ops in recorded order)
        if self.binary_specs_:
            new_cols = []                                           # will collect new binary columns
            for (i, j, op_name, new_name) in self.binary_specs_:    # iterate kept binaries
                vi = U[:, i]                                        # left operand values
                vj = U[:, j]                                        # right operand values
                v = BINARY_OPS[op_name](vi, vj)                     # apply binary op
                new_cols.append(v)                                  # collect
                cols.append(new_name)                               # append column name
            U = np.column_stack([U, np.column_stack(new_cols)]) if new_cols else U  # append to base

        return pd.DataFrame(U, columns=cols)                        # final stage output DataFrame
