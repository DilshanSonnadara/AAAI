from typing import Iterable, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):  
    """
    Categorical encoder with an optional explicit categorical list
    using FREQUENCY ENCODING only.

    Behavior
    --------
    - Columns listed in `categorical_cols` are treated as categorical
      and encoded using relative frequencies. However, if they have a numeric nature, their original values are preserved.
    - Other columns are inferred:
        * If all non-missing values are numeric, the column is treated as numeric
          and passed through as float with NaNs imputed using the median.
        * Otherwise, the column is treated as categorical and frequency-encoded.

    Encoding details
    ---------------
    - Categorical values are encoded as relative frequencies only.
    - During `fit`, missing values are imputed (mode for categorical, median for numeric).
    - During `transform`, the SAME learned imputations are applied.
    - Unseen categories are encoded as `freq_unknown_value`.

    Attributes
    ----------
    feature_types_ : dict[str, str]
        Column -> {"numeric","categorical"} inferred during fit.
    freq_maps_ : dict[str, dict]
        Per categorical column: category -> relative frequency.
    numeric_imputers_ : dict[str, float]
        Per numeric column: median value.
    numeric_like_categoricals_ : set[str]
        Explicit categorical columns detected as numeric-like and preserved as numeric.
    categorical_imputers_ : dict[str, str]
        Per categorical column: mode value.
    fitted_ : bool
        Whether `fit()` has been called.
    """

    def __init__(
        self,
        *, # The * means everything after it MUST be passed by name
        freq_unknown_value: float = 0.0, 
        categorical_cols: Optional[Iterable[str]] = None, # something you can loop over that contains strings(list, set, tuple, etc.)
    ):                                                    # Optional[...] → either that type or None,  = None → default value is None
        """
        Initialize the encoder.
        """
        self.freq_unknown_value = float(freq_unknown_value)  # value for unseen categories
        self.categorical_cols = set(categorical_cols or [])  # explicit categorical columns

        self.freq_maps_: dict[str, dict] = {}                # category -> relative frequency
        self.numeric_imputers_: dict[str, float] = {}        # column -> median
        self.categorical_imputers_: dict[str, str] = {}     # column -> mode
        self.feature_types_: dict[str, str] = {}             # column -> type
        self.numeric_like_categoricals_: set[str] = set()
        self.fitted_: bool = False                            # fit flag

    def _is_numeric_allowing_nans(self, s: pd.Series) -> bool:
        """
        Check if a column is numeric-like (ignoring missing values).
        """
        s_no_na = s.dropna()  # drop missing values
        if s_no_na.empty:     # all missing → treat as numeric
            return True
        return pd.to_numeric(s_no_na, errors="coerce").notna().all() #Tries to convert every value to a number, 
                                                                     #If a value cannot be converted → it becomes NaN
                                                                     #Checks which values are NOT NaN
                                                                     #Returns True only if ALL values are True
    def _validate_columns(self, df: pd.DataFrame) -> None:  
        """
        Ensure user-specified categorical columns exist.
        """
        missing = self.categorical_cols - set(df.columns)
        if missing:
            raise ValueError(f"Categorical columns not found: {sorted(missing)}")

    def fit(self, X, y =None) -> "CategoricalEncoder":
        """
        Learn imputations, frequency encodings, and feature types.
        """
        df = pd.DataFrame(X).copy()
        self._validate_columns(df)

        #These four lines reset the encoder’s memory before learning from new data.
        self.freq_maps_.clear()
        self.numeric_imputers_.clear()
        self.categorical_imputers_.clear()
        self.feature_types_.clear()
        self.numeric_like_categoricals_.clear()

        for col in df.columns:
            s = df[col]

            if col in self.categorical_cols:
                # NEW: check if it's actually numeric-like
                if self._is_numeric_allowing_nans(s):
                    # Treat as numeric instead
                    s_num = pd.to_numeric(s, errors="coerce")
                    self.numeric_imputers_[col] = s_num.median()
                    self.feature_types_[col] = "categorical"
                    self.numeric_like_categoricals_.add(col)   # 🔥 key line


                else:
                    # Explicit categorical → mode imputation + frequency encoding
                    s_str = s.astype("string")

                    mode = s_str.dropna().mode()
                    self.categorical_imputers_[col] = mode.iloc[0] if not mode.empty else "__MISSING__"

                    freqs = (
                        s_str.fillna(self.categorical_imputers_[col])
                        .value_counts(normalize=True) #Counts how often each value appears,  Divides by the total number of values
                                                    # Returns relative frequencies (proportions) instead of raw counts
                    )

                    self.freq_maps_[col] = freqs.to_dict()
                    self.feature_types_[col] = "categorical"

            elif self._is_numeric_allowing_nans(s):
                # Numeric → median imputation
                s_num = pd.to_numeric(s, errors="coerce") #Tries to convert every value in s into a number,  If a value cannot be converted → it becomes NaN
                self.numeric_imputers_[col] = s_num.median()
                self.feature_types_[col] = "numeric"

            else:
                # Inferred categorical → mode imputation + frequency encoding
                s_str = s.astype("string")

                mode = s_str.dropna().mode()
                self.categorical_imputers_[col] = mode.iloc[0] if not mode.empty else "__MISSING__"

                freqs = (
                    s_str.fillna(self.categorical_imputers_[col])
                    .value_counts(normalize=True)
                )

                self.freq_maps_[col] = freqs.to_dict()
                self.feature_types_[col] = "categorical"

        self.fitted_ = True
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Apply learned imputations and frequency encodings.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        df = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=df.index)

        for col in df.columns:
            ftype = self.feature_types_.get(col, "numeric") # If the column was NOT seen, it defaults to "numeric"

            if ftype == "numeric" or col in self.numeric_like_categoricals_:                # Numeric passthrough with median imputation
                median = self.numeric_imputers_.get(col, 0.0)
                out[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(median)
                    .astype(float)
                )
            else:
                # Categorical mode imputation + frequency encoding
                fill_value = self.categorical_imputers_.get(col, "__MISSING__")
                freq_map = self.freq_maps_.get(col, {}) #Tries to get the frequency map for this column

                out[col] = (
                    df[col].astype("string") #Converts values to pandas’ string type
                    .fillna(fill_value) #Replaces missing values, So categorical NaNs → mode
                    .map(freq_map) #Replaces each category with its relative frequency (Known category → its frequency,Unknown category → NaN)
                    .fillna(self.freq_unknown_value) #Handles unseen categories, Any value not in freq_map → freq_unknown_value
                    .astype(float) #Ensures the final column is numeric
                )

        return out

    def fit_transform(self, X,y=None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        return self.fit(X,y).transform(X)

    def get_freq_maps(self) -> dict[str, dict]:
        """
        Return learned frequency maps.
        """
        return {k: dict(v) for k, v in self.freq_maps_.items()}

    def get_feature_types(self) -> dict[str, str]:
        """
        Return inferred feature types.
        """
        return dict(self.feature_types_)
    
