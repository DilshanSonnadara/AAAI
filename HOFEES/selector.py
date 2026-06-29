from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Callable, Dict, Any, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.inspection import permutation_importance

from .metrics import score_with_target, _distance_correlation
from sklearn.model_selection import train_test_split

# ============================================================
# Model-aware selector used inside the workflow
# ============================================================

@dataclass
class _Selector:
    """
    Lightweight **model-aware feature selector**:
      - Ranks columns by score with the target,
      - Sweeps a grid of k (top-k columns) on a validation split,
      - Picks the k that maximizes validation performance.,
      - Fits and stores the final model on **all data** with the chosen columns.

    Parameters
    ----------
    model : sklearn BaseEstimator
        The estimator to fit/evaluate (e.g., LogisticRegression, RandomForest, etc.).
    task : {"classification","regression"}
        Controls score computation inside ranking.
    metric : str or callable
        Either an sklearn scorer name (e.g., "roc_auc", "r2") or a callable scorer(est, X, y).
    iterations : int, default=20
        How many k steps to evaluate (grid = [k_step, 2*k_step, ..., iterations*k_step]).
    k_step : int, default=5
        Step size for k in the sweep.
    val_size : float, default=0.2
        Fraction of the data used as validation split.
    random_state : int, default=42
        Seed for the train/validation split.

    Attributes
    ----------
    k_grid_ : List[int] or None
        The realized k values that were evaluated.
    val_curve_ : List[Tuple[int, float]] or None
        Pairs of (k, validation_score) collected during sweep.
    best_k_ : int or None
        The chosen k with best validation score.
    best_cols_ : List[str] or None
        The top-k column names used for the final model.
    final_model_ : BaseEstimator or None
        The model refit on the **entire** stage matrix with selected columns.
    score_scores_ : List[Tuple[str, float]]
        Features ranked by their dependency or permutation score.

    ranked_cols_ : List[str]
        Feature names ordered from highest to lowest score.
    """

    model: BaseEstimator                                         # estimator to evaluate
    task: str                                                    # task type
    metric: Union[str, Callable]                                 # scoring spec
    iterations: Optional[int] = 20                               # number of k steps
    k_step: int = 5                                              # stride in k
    val_size: float = 0.2                                        # validation fraction
    random_state: Optional[int] = 42                             # split seed
    fs_score_mode: Literal["dependency", "permutation"] = "dependency"  # ranking strategy. Dependency uses
                                                                        # distance correlation

    # outputs
    k_grid_: Optional[List[int]] = None                          # evaluated k values
    val_curve_: Optional[List[Tuple[int, float]]] = None         # (k, score) pairs
    best_k_: Optional[int] = None                                # best k
    best_cols_: Optional[List[str]] = None                       # chosen columns
    final_model_: Optional[BaseEstimator] = None                 # final refit model
    dep_threshold: float = 0.6                                   # keep only strongly dependent features
    _pi_repeats: int = field(init=False, default=8)              # permutation importance repeats
    feature_types: Optional[Dict[str, str]] = None               # optional column -> type map for scoring

    def _build_scorer(self):
        """Return a scorer callable from `metric` (sklearn scorer name or callable)."""
        if isinstance(self.metric, str):                         # metric given by name
            return get_scorer(self.metric)                       # obtain scorer from sklearn
        if callable(self.metric):                                # already a callable
            return self.metric                                   # return as-is
        raise ValueError("metric must be sklearn scorer name or callable scorer.")


    def _rank_by_dependency(self, X: pd.DataFrame, y_arr: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank features by dependency score (distance correlation for both tasks).
        - Features with dependency scores below `dep_threshold` are discarded before ranking.
        - Features containing non-finite or extremely large values are excluded from ranking.
        """
        MAX_F32 = np.finfo(np.float32).max  # ~3.4e38
        score_scores: List[Tuple[str, float]] = []
        for col in X.columns:
            try:
                arr = X[col].to_numpy()
                finite = np.isfinite(arr)
                if not finite.all() or np.any(np.abs(arr[finite]) > MAX_F32):
                    continue


                if self.task == "classification":
                    s = _distance_correlation(
                        X[col].to_numpy(),
                        y_arr,
                    )
                else:
                    s = score_with_target(
                        X[col].to_numpy(),
                        y_arr,
                        task=self.task,
                    )
                if s < self.dep_threshold:
                    continue

                score_scores.append((col, float(s)))
            except Exception:
                continue
        score_scores.sort(key=lambda t: t[1], reverse=True)
        return score_scores
    



    MAX_F32 = np.finfo(np.float32).max  # ~3.4e38

    # inside _Selector
    def _split_good_bad(self, X: pd.DataFrame, max_abs: float = MAX_F32):
        good, bad = [], []
        for col in X.columns:
            arr = X[col].to_numpy()
            finite = np.isfinite(arr)
            if not finite.all() or np.any(np.abs(arr[finite]) > max_abs):
            # if not finite.all():

                bad.append(col)
            else:
                good.append(col)
        return X[good], bad

    def _rank_by_permutation(self, X: pd.DataFrame, y_arr: np.ndarray):
        scorer = self._build_scorer()
        X_clean, dropped = self._split_good_bad(X)
        if X_clean.shape[1] == 0:
            # no usable columns; return dropped ones with zero importance
            return [(c, 0.0) for c in dropped]

        model_fit = clone(self.model).fit(X_clean, y_arr)
        result = permutation_importance(
            model_fit,
            X_clean,
            y_arr,
            scoring=scorer,
            n_repeats=self._pi_repeats,
            random_state=self.random_state,
        )
        pairs = list(zip(X_clean.columns, result.importances_mean.tolist()))
        # append dropped columns with zero importance
        # pairs.extend((c, 0.0) for c in dropped)
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs


    def run(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        feature_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute score ranking + k-sweep selection + final refit.

        Steps
        -----
        1) Compute a score for each column and rank descending.
        2) Train/validation split.
        3) For k in increasing steps, fit model on top-k columns and score on validation set.
        4) Keep best k, refit final model on **all** X (with best columns).

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, d)
            Stage matrix (columns are engineered features of this stage).
        y : array-like of shape (n_samples,)
            Target.
        - If no features pass the filtering step, an empty result is returned.

        Returns
        -------
        Dict[str, Any]
            A summary with keys:
              - "k_grid": List[int]
              - "val_curve": List[Tuple[int, float]]
              - "best_k": int
              - "selected_columns": List[str]
              "score_ranking"
              "ranked_columns"
        """
        y_arr = np.asarray(y).reshape(-1)                         # vectorize target
        self.feature_types = feature_types
    
        if self.fs_score_mode == "permutation":
            score_scores = self._rank_by_permutation(X, y_arr)
        else:
            score_scores = self._rank_by_dependency(X, y_arr)
        ranked_cols = [c for c, _ in score_scores]


        # --- Handle case: no features survived ---
        if len(ranked_cols) == 0:
            msg = (
                "No useful features or interactions identified"
            )
            print(msg)
            # Return clean summary
            return {
                "k_grid": [],
                "val_curve": [],
                "best_k": 0,
                "selected_columns": [], 
                "score_ranking": [],
                "ranked_columns": [],
            }


        # ADD:
        self.score_scores_ = list(score_scores)           # list of (column, score)
        self.ranked_cols_ = list(ranked_cols)       # columns ordered by score
        d = len(ranked_cols)                                      # total columns

        

        # Split into sub-train and validation
        X_subtr, X_val, y_subtr, y_val = train_test_split(
            X, y_arr, test_size=self.val_size, random_state=self.random_state,
            stratify=y_arr if self.task == "classification" else None  # stratify only for classification
        )

        scorer = self._build_scorer()                             # obtain scorer callable

        
        

        # Build k grid (bounded by d); ensure at least one k exists
        if self.iterations is None:
            # Sweep until all features are covered
            raw_grid = list(range(self.k_step, d + 1, self.k_step))
        else:
            raw_grid = [self.k_step * i for i in range(1, self.iterations + 1)]  # k candidates

        self.k_grid_ = sorted(set([k for k in raw_grid if k <= d])) or [min(self.k_step, d)]  # clamp to d


        self.val_curve_ = []                                      # will collect (k, score)

        best_score = -np.inf                                       # track best score
        best_k = None                                              # track best k
        best_cols = None                                           # track best column list

        for k in self.k_grid_:                                     # iterate k sweep
            cols_k = ranked_cols[:k]                               # top-k columns
            m_k = clone(self.model)                                # fresh model
            m_k.fit(X_subtr[cols_k], y_subtr)                      # fit on sub-train

            # score via sklearn scorer (uses predict/predict_proba/etc.)
            score = float(scorer(m_k, X_val[cols_k], y_val))
            self.val_curve_.append((k, score))                     # record validation point
            if score > best_score:                                 # update best
                best_score, best_k, best_cols = score, k, cols_k

        self.best_k_ = int(best_k) if best_k is not None else len(ranked_cols)  # finalize best k
        self.best_cols_ = list(best_cols) if best_cols is not None else ranked_cols  # finalize best cols

        # Refit final model on **all** rows with selected columns
        self.final_model_ = clone(self.model).fit(X[self.best_cols_], y_arr)

        return {
            "k_grid": self.k_grid_,
            "val_curve": self.val_curve_,
            "best_k": self.best_k_,
            "selected_columns": self.best_cols_,
            # ADD:
            "score_ranking": self.score_scores_,          # [(col, score), ...] desc by score
            "ranked_columns": self.ranked_cols_,    # [col, ...] desc by score
        }
