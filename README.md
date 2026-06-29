# HOFEES

This repository contains the anonymous implementation of **HOFEES (Higher-Order Automated Feature Engineering through Exhaustive Search for Lower Dimensional Feature Subspaces)** together with the datasets and experiment scripts used in the accompanying AAAI submission.

## Overview

HOFEES is an automated feature engineering framework that progressively constructs higher-order feature representations through exhaustive exploration of low-dimensional feature subspaces. The framework combines unary feature transformations, binary feature interactions, and model-aware feature selection to generate compact and informative feature spaces for machine learning.

## Example Usage

from sklearn.ensemble import RandomForestRegressor
from HOFEES.workflow import AutoFEWorkflow

wf = AutoFEWorkflow(
    task="regression",
    model=RandomForestRegressor(random_state=42),
    metric="neg_root_mean_squared_error",
    iterations=None,
    k_step=1,
    score_binary=0.0,
    score_eps=0.0,
    dep_threshold=0.0,
    n_interaction_levels=1,
    random_state=42,
    model_aware_fs_mode="each_stage",
    encode_categoricals=False
)

wf.run(X_train, y_train)

X_train_transformed = wf.transform_selected(X_train)
X_test_transformed = wf.transform_selected(X_test)

The transformed datasets can now be used to train a ML model

## Citation

If this work is accepted for publication, the citation will be added here.

## License

This repository is released under the MIT License.
