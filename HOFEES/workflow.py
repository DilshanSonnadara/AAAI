from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from .stage import AutoFEStage
from .selector import _Selector
from .encoding import CategoricalEncoder
from .metrics import score_with_target, _distance_correlation
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# Multi-stage Workflow
#   Stage 1: full-unary + 2-way interactions -> FS (model-aware)
#   Stage 2..L: full-unary (again) + constrained pairs -> FS each stage
# ============================================================

@dataclass
class AutoFEWorkflow(BaseEstimator, TransformerMixin):
    """
    Multi-stage **score-Aware AutoFE** pipeline:

      • Stage 1:
        - full unary search + 2-way interaction mining (no growth constraint)
        - model-aware selection (optional) (_Selector) over the stage outputs

      • Stage 2..L:
        - full unary search again on the *selected* features from the previous stage
        - 2-way interaction mining but **require at least one operand to be newly-created** in the prior stage
          (enforces growth into higher-order interactions across stages)
        - model-aware selection per stage (if selected)

      • Final:
        - The last stage’s selector’s model is stored as `final_model_`.
        - `transform_selected(X)` converts new inputs to the final selected matrix.
        - `predict`/`predict_proba` apply the final model.


    Parameters
    ----------
    task : {"classification","regression"}, default="classification"
        Downstream task type. Controls scoring functions and model behaviour.

    n_interaction_levels : int, default=1
        Number of feature engineering stages.
        - 1 → up to second-order interactions
        - 2 → up to third-order interactions
        Higher values progressively build higher-order interactions.

    model : BaseEstimator, default=RandomForestClassifier
        Estimator used for model-aware feature selection.

    metric : str or callable, default="roc_auc"
        Validation metric used during feature selection.
        Can be a sklearn scorer name or a callable.

    iterations : int, default=20
        Number of k values evaluated during feature selection (k-sweep).

    k_step : int, default=5
        Step size for k in the selection sweep.

    random_state : int, default=42
        Random seed used for reproducability.

    encode_categoricals : bool, default=True
        If True, categorical features are encoded using frequency encoding
        before feature engineering.

    categorical_cols : list[str] or None, default=None
        List of columns explicitly treated as categorical.
        If None, categorical features are inferred automatically.

    score_eps : float, default=0.0
        Minimum improvement required for a unary transformation to replace
        the identity transformation.

    score_binary : float, default=0.0
        Minimum improvement required for a binary feature to be retained
        over its individual operands.

    dep_threshold : float, default=0.0
        Minimum dependency score (distance correlation) required for a feature
        to be considered during ranking.

    fs_score_mode : {"dependency", "permutation"}, default="dependency"
        Feature ranking strategy:
        - "dependency": uses distance correlation with the target
        - "permutation": uses permutation importance

    model_aware_fs_mode : {"each_stage","final_only","off"}, default="final_only"
        Controls when model-aware feature selection is applied:
        - "each_stage": selection at every stage
        - "final_only": selection only at the final stage
        - "off": no model-aware selection

    stage_ranked_cols_ : List[List[str]]
        Ranked feature names for each stage.

    stage_score_ranking_ : List[List[Tuple[str, float]]]
        Feature scores for each stage in descending order.

    Notes
    -----
    Internal fixed constants (not user-tunable): val_size=0.2,
    keep_n_per_stage="orig", unknown_category_value=0.

    Attributes
    ----------
    stages_ : List[AutoFEStage]
        Fitted FE stages in order.
    stage_selected_cols_ : List[List[str]]
        For each stage, the selected columns (names) passed to the next stage.
    final_model_ : BaseEstimator or None
        The final fitted model from the **last stage’s** selector.
    """

    # FE knobs
    task: str = "classification"                                  # task type
    n_interaction_levels: int = 1                                 # number of stages

    # FS knobs
    model: BaseEstimator = field(default_factory=lambda: RandomForestClassifier(random_state=42))  # default estimator
    metric: Union[str, Callable] = "roc_auc"                       # validation scorer
    iterations: int = 20                                           # k sweep length
    k_step: int = 5                                                # k stride
    val_size: float = field(init=False, default=0.2)  # fixed validation fraction
    random_state: Optional[int] = 42                             # reproducibility

    # Artifacts
    stages_: List[AutoFEStage] = field(init=False, default_factory=list)    # fitted stages
    stage_selected_cols_: List[List[str]] = field(init=False, default_factory=list)# per-stage selected colnames
    final_model_: Optional[BaseEstimator] = field(init=False, default=None)        # final model handle
    stage_ranked_cols_: List[List[str]] = field(init=False, default_factory=list)
    stage_score_ranking_: List[List[tuple]] = field(init=False, default_factory=list)

    # ---- Encoding knobs ----
    encode_categoricals: bool = True  # Applied for both tasks when True
    unknown_category_value: float = field(init=False, default=0.0)  # fill value for unseen categories
    categorical_cols: Optional[list[str]] = None  # user-provided categorical columns

    # Hyperparameters surfaced at the workflow level
    score_eps: float = 0.0
    score_binary: float = 0.0
    dep_threshold: float = 0.0
    fs_score_mode: Literal["dependency", "permutation"] = "dependency"

    # ---- Internals ----
    _cat_encoder: "CategoricalEncoder | None" = field(init=False, default=None)

    # --- New knobs ---
    model_aware_fs_mode: Literal["each_stage","final_only","off"] = "final_only"
    keep_n_per_stage: Union[int, Literal["orig"]] = field(init=False, default="orig")  # fixed per-stage keep rule
    # “keep as many features as the original input had” in each stage of the “top‑n per stage” branch of run()

    def _rank_top_k(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        k: int,
        *,
        feature_types: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """
        Rank columns in X using the configured fs_score_mode (dependency or permutation) and return
        top-k names plus the (name, score) list. Dependency mode drops scores below dep_threshold.
        """
        y_arr = np.asarray(y).reshape(-1)
        selector = _Selector(
            model=self.model,
            task=self.task,
            metric=self.metric,
            iterations=self.iterations,
            k_step=self.k_step,
            val_size=self.val_size,
            random_state=self.random_state,
            dep_threshold=self.dep_threshold,
            fs_score_mode=self.fs_score_mode,
        )

        if self.fs_score_mode == "permutation":
            scored = selector._rank_by_permutation(X, y_arr)
        else:
            scored = selector._rank_by_dependency(X, y_arr)

        if len(scored) == 0:
            return [], []
        k = max(1, min(k, len(scored)))
        top_names = [c for c, _ in scored[:k]]
        return top_names, scored


    def _fit_stage_and_select(
        self,
        X_in: pd.DataFrame,                                        # input matrix for this stage
        y: np.ndarray,                                             # target
        *,
        unary_mode: str,                                           # "full" or "identity"
        require_prev_new: bool,                                    # enforce growth constraint?
        prev_new_mask: Optional[np.ndarray],                       # previously-new mask (aligned to X_in columns)
        feature_types_in: Optional[dict[str, str]],                # column -> type for guarding transforms
    ) -> Tuple[AutoFEStage, pd.DataFrame, List[str], np.ndarray, dict[str, str]]:
        """
        Fit one FE stage on X_in **and** perform selection over that stage’s outputs.

        Returns
        -------
        stage : AutoFEStage
            The fitted stage object (contains unary/binary choices).
        X_next : pd.DataFrame
            The **selected** columns of the stage output to feed into the next stage.
        selected_cols : List[str]
            Names of selected columns at this stage.
        new_binary_mask_next : np.ndarray[bool]
            Mask aligned to `X_next.columns` marking which selected columns are **new binaries**
            (used to enforce growth in the following stage).
        """
        stage = AutoFEStage(                                # instantiate a stage
            task=self.task,
            random_state=self.random_state,
            score_eps=self.score_eps,            # NEW
            score_binary=self.score_binary,      # NEW
        ).fit(
            X_in, y,                                               # fit stage on current input
            require_prev_new=require_prev_new,                     # growth constraint flag
            prev_new_mask=prev_new_mask,                           # previous new mask for pairs
            unary_mode=unary_mode,                                 # "full"/"identity" unary behavior
            feature_types=feature_types_in,
        )

        X_stage = stage.transform(X_in)                            # produce full stage outputs (base + binaries)

        # Propagate feature types through this stage: base cols keep their input type, binaries are numeric
        base_types = [feature_types_in.get(col, "numeric") if feature_types_in else "numeric" for col in X_in.columns]
        types_full = base_types.copy()
        if stage.binary_specs_:
            types_full.extend(["numeric"] * len(stage.binary_specs_))
        ft_map_stage = {col: t for col, t in zip(X_stage.columns, types_full)}

        # Model-aware selector on this stage’s outputs
        selector = _Selector(
            model=self.model,
            task=self.task,
            metric=self.metric,
            iterations=self.iterations,
            k_step=self.k_step,
            val_size=self.val_size,
            random_state=self.random_state,
            dep_threshold=self.dep_threshold,
            fs_score_mode=self.fs_score_mode,
        )
        sel_summary = selector.run(X_stage, y, feature_types=ft_map_stage)                     # run score ranking + k sweep + final refit
        selected_cols = sel_summary["selected_columns"]            # chosen column names
        # ADD:
        ranked_cols = sel_summary["ranked_columns"]
        score_ranking  = sel_summary["score_ranking"]

        # record per-stage score ranking
        self.stage_ranked_cols_.append(list(ranked_cols))
        self.stage_score_ranking_.append(list(score_ranking))

        # Build next-stage input from selected columns only
        X_next = X_stage[selected_cols].copy()                     # selected sub-matrix

        feature_types_next = {c: ft_map_stage.get(c, "numeric") for c in selected_cols}

        # Convert stage.new_binary_mask_ (aligned to X_stage.columns) into mask for X_next.columns
        if stage.new_binary_mask_ is None or len(stage.new_binary_mask_) != X_stage.shape[1]:
            new_mask_next = np.zeros(len(selected_cols), dtype=bool)  # fallback: none marked
        else:
            name_to_isnew = {col_name: bool(stage.new_binary_mask_[idx]) for idx, col_name in enumerate(X_stage.columns)}
            new_mask_next = np.array([name_to_isnew.get(c, False) for c in selected_cols], dtype=bool)

        # Store this stage's final model as **current** final model (updated at each stage)
        self.final_model_ = selector.final_model_

        return stage, X_next, selected_cols, new_mask_next, feature_types_next

    def run(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Execute the full multi-stage FE + model-aware FS workflow.

        Stage 1:
            - full unary search + 2-way interactions (no growth constraint)
            - selection

        Stage 2..L:
            - full unary search again on previous **selected** matrix
            - 2-way interactions **requiring at least one operand to be previously-new**
            - selection

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training matrix.
        y_train : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        Dict[str, Any]
            - "n_stages": int
            - "stage_selected_columns": List[List[str]]
            - "final_selected_columns": List[str]
            - "final_model": BaseEstimator
        """
        X_current = pd.DataFrame(X_train)                          # normalize to DataFrame
        y_arr = np.asarray(y_train).reshape(-1)                    # vectorize target

        # --- Encode categoricals (applies to both tasks when enabled) and capture feature types ---
        if self.encode_categoricals:
            if self._cat_encoder is None:
                # Preserve user-specified categorical column order/mask; avoid set/sort
                cat_cols = list(self.categorical_cols or [])
                self._cat_encoder = CategoricalEncoder(
                    freq_unknown_value=self.unknown_category_value,
                    categorical_cols=cat_cols,
                )
            X_current = self._cat_encoder.fit_transform(X_current)
            feature_types_current = self._cat_encoder.get_feature_types()
        else:
            feature_types_current = {c: "numeric" for c in X_current.columns}

        self.stages_.clear()                                       # reset stages list
        self.stage_selected_cols_.clear()                          # reset selections log
        self.final_model_ = None                                   # clear final model
        self.stage_ranked_cols_.clear()
        self.stage_score_ranking_.clear()

        # how many to keep per stage
        orig_n = X_current.shape[1]
        n_keep = orig_n if self.keep_n_per_stage == "orig" else int(self.keep_n_per_stage)
        if self.model_aware_fs_mode == "each_stage":
            # Stage 1: full unary + 2-way interactions (no growth constraint)
            stage1, X_next, sel_cols1, new_mask_next, feature_types_next = self._fit_stage_and_select(
                X_current, y_arr,
                unary_mode="full",                                     # full search on base columns
                require_prev_new=False,                                # allow any pairs at level 1
                prev_new_mask=None,                                     # not applicable
                feature_types_in=feature_types_current,
            )
            self.stages_.append(stage1)                                # remember stage object
            self.stage_selected_cols_.append(sel_cols1)                # remember selected columns
            feature_types_current = feature_types_next

            # Subsequent stages (2..L): full unary + constrained pairs (at least one is previously-new)
            for level in range(2, self.n_interaction_levels + 1):      # iterate remaining levels
                stage_k, X_next, sel_cols_k, new_mask_next, feature_types_next = self._fit_stage_and_select(
                    X_next, y_arr,
                    unary_mode="full",                                 # full unary again (strong search)
                    require_prev_new=True,                             # enforce growth into higher-order interactions
                    prev_new_mask=new_mask_next,                       # mask aligned to X_next
                    feature_types_in=feature_types_current,
                )
                self.stages_.append(stage_k)                           # append stage
                self.stage_selected_cols_.append(sel_cols_k)           # append selected columns
                feature_types_current = feature_types_next

            return {
                "n_stages": len(self.stages_),                         # total number of stages fitted
                "stage_selected_columns": self.stage_selected_cols_,    # per-stage selections
                "final_selected_columns": self.stage_selected_cols_[-1] if self.stage_selected_cols_ else [],  # last stage
                "final_model": self.final_model_,                      # final fitted model handle
                # ADD:
                "stage_ranked_columns": self.stage_ranked_cols_,   # per-stage score order (names only)
                "stage_score_ranking": self.stage_score_ranking_,        # per-stage [(name, score), .
            }

        # ===== Branch 2: New flow (keep top-n each stage) =====
        # Stage 1: fit + transform
        stage1 = AutoFEStage(
            task=self.task,
            random_state=self.random_state,
            score_eps=self.score_eps,
            score_binary=self.score_binary,
        ).fit(
            X_current, y_arr,
            require_prev_new=False,
            prev_new_mask=None,
            unary_mode="full",
            feature_types=feature_types_current,
        )
        X_stage = stage1.transform(X_current)
        self.stages_.append(stage1)

        # feature type propagation for stage outputs
        base_types = [feature_types_current.get(col, "numeric") for col in X_current.columns]
        types_full = base_types.copy()
        if stage1.binary_specs_:
            types_full.extend(["numeric"] * len(stage1.binary_specs_))
        ft_map_stage1 = {col: t for col, t in zip(X_stage.columns, types_full)}

        # Rank & keep top-n for Stage 1
        names1, ranked1 = self._rank_top_k(X_stage, y_arr, n_keep, feature_types=ft_map_stage1)
        self.stage_ranked_cols_.append([c for c, _ in ranked1])
        self.stage_score_ranking_.append(list(ranked1))
        X_next = X_stage[names1].copy()
        feature_types_current = {c: ft_map_stage1.get(c, "numeric") for c in names1}
        # build "new binary" mask for the selected subset to enforce growth at next stage
        if stage1.new_binary_mask_ is not None and len(stage1.new_binary_mask_) == X_stage.shape[1]:
            name_to_isnew = {col_name: bool(stage1.new_binary_mask_[idx]) for idx, col_name in enumerate(X_stage.columns)}
            new_mask_next = np.array([name_to_isnew.get(c, False) for c in names1], dtype=bool)
        else:
            new_mask_next = np.zeros(len(names1), dtype=bool)
        self.stage_selected_cols_.append(names1)

        # Subsequent stages: full unary + constrained pairs, then keep top-n
        for level in range(2, self.n_interaction_levels + 1):
            stage_k_input_cols = list(X_next.columns)
            stage_k = AutoFEStage(
                task=self.task,
                random_state=self.random_state,
                score_eps=self.score_eps,
                score_binary=self.score_binary,
            ).fit(
                X_next, y_arr,
                require_prev_new=True,
                prev_new_mask=new_mask_next,
                unary_mode="full",
                feature_types=feature_types_current,
            )
            X_stage = stage_k.transform(X_next)
            self.stages_.append(stage_k)

            base_types_k = [feature_types_current.get(col, "numeric") for col in stage_k_input_cols]
            types_full_k = base_types_k.copy()
            if stage_k.binary_specs_:
                types_full_k.extend(["numeric"] * len(stage_k.binary_specs_))
            ft_map_stage_k = {col: t for col, t in zip(X_stage.columns, types_full_k)}

            names_k, ranked_k = self._rank_top_k(X_stage, y_arr, n_keep, feature_types=ft_map_stage_k)
            self.stage_ranked_cols_.append([c for c, _ in ranked_k])
            self.stage_score_ranking_.append(list(ranked_k))
            X_next = X_stage[names_k].copy()
            feature_types_current = {c: ft_map_stage_k.get(c, "numeric") for c in names_k}

            if stage_k.new_binary_mask_ is not None and len(stage_k.new_binary_mask_) == X_stage.shape[1]:
                name_to_isnew = {col_name: bool(stage_k.new_binary_mask_[idx]) for idx, col_name in enumerate(X_stage.columns)}
                new_mask_next = np.array([name_to_isnew.get(c, False) for c in names_k], dtype=bool)
            else:
                new_mask_next = np.zeros(len(names_k), dtype=bool)
            self.stage_selected_cols_.append(names_k)

        # ===== Finalization depending on model_aware_fs_mode =====
        if self.model_aware_fs_mode == "final_only":
            # run selector ONCE on the last stage's full matrix using the current X_next (which is already top-n)
            selector = _Selector(
                model=self.model,
                task=self.task,
                metric=self.metric,
                iterations=self.iterations,
                k_step=self.k_step,
                val_size=self.val_size,
            random_state=self.random_state,
            dep_threshold=self.dep_threshold,
            fs_score_mode=self.fs_score_mode,
        )
            sel_summary = selector.run(X_next, y_arr, feature_types=feature_types_current)
            self.final_model_ = selector.final_model_
            final_cols = sel_summary["selected_columns"]
            # Replace last stage selection with the model-aware pick (log both orders)
            self.stage_selected_cols_[-1] = list(final_cols)
            self.stage_ranked_cols_[-1] = list(sel_summary["ranked_columns"])
            self.stage_score_ranking_[-1] = list(sel_summary["score_ranking"])

        else:  # "off" + no model-aware FS at all
            self.final_model_ = None  # nothing fitted at the end

        return {
            "n_stages": len(self.stages_),
            "stage_selected_columns": self.stage_selected_cols_,
            "final_selected_columns": self.stage_selected_cols_[-1] if self.stage_selected_cols_ else [],
            "final_model": self.final_model_,
            "stage_ranked_columns": self.stage_ranked_cols_,
            "stage_score_ranking": self.stage_score_ranking_,
        }

    # ---------- Inference helpers ----------

    def _transform_through_stages(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Apply all fitted stages in sequence to produce the **final selected design matrix**.

        Each stage:
          - transforms X to the stage’s full output,
          - then keeps **only** the columns that were selected at training time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data in the same format/column order as during training.

        Returns
        -------
        pd.DataFrame
            Final selected feature matrix ready for the final model.
        Code - Verified
        """
        if not self.stages_:                                       # ensure workflow is fitted
            raise RuntimeError("Workflow is not fitted. Call run() first.")
        X_current = pd.DataFrame(X)                                # normalize input
        # --- NEW: apply the SAME train-learned mapping on INFERENCE ---
        if self.encode_categoricals and self._cat_encoder is not None:
            X_current = self._cat_encoder.transform(X_current)

        # Running the rest of the workflow
        for stage, selected_cols in zip(self.stages_, self.stage_selected_cols_):  # walk stages in order
            X_stage = stage.transform(X_current)                   # produce stage output
            X_current = X_stage[selected_cols].copy()              # keep only selected cols

        return X_current                                           # final selected matrix

    def transform_selected(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform new data **into the final selected engineered feature matrix**.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pd.DataFrame
            Matrix with exactly the columns selected by the final workflow.
        Code - Verified
        """
        return self._transform_through_stages(X)                   # delegate to internal helper

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict with the final model after transforming into the selected feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions from `final_model_`.
        Code - Verified
        """
        if self.final_model_ is None:                              # check fitted
            raise RuntimeError("No final model. Run the workflow first.")
        X_sel = self.transform_selected(X)                         # transform into final selected features
        return self.final_model_.predict(X_sel)                    # call estimator predict

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Class probabilities with the final model (classification tasks only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Class probability estimates from `final_model_`.

        Raises
        ------
        AttributeError
            If the final model does not implement `predict_proba`.
        Code - Verified
        """
        if self.final_model_ is None:                              # ensure fitted
            raise RuntimeError("No final model. Run the workflow first.")
        if not hasattr(self.final_model_, "predict_proba"):        # ensure capability
            raise AttributeError("Final model has no predict_proba.")
        X_sel = self.transform_selected(X)                         # transform into selected feature space
        return self.final_model_.predict_proba(X_sel)              # call estimator predict_proba


    def fit(self, X, y):
        self.run(X, y)   # call your existing logic
        self.fitted_ = True
        return self
    
    def transform(self, X):
        return self._transform_through_stages(X)