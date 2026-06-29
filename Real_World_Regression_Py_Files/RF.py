#Import the necessary libraries

import pickle
import os
import time
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from dcor import distance_correlation
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from HOFEES.workflow import AutoFEWorkflow
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

#Open the data files
BASE_DIR = Path(__file__).resolve().parent

file_path = (
    BASE_DIR
    / ".."
    / "Data"
    / "All Data"
    / "Cleaned Data"
    / "Dependent_Data_dictionary.pkl"
).resolve()

file_path_Independent = (
    BASE_DIR
    / ".."
    / "Data"
    / "All Data"
    / "Cleaned Data"
    / "Independent_Data_dictionary.pkl"
).resolve()

with open(file_path, "rb") as f:
    Dependent_Data_dictionary = pickle.load(f)


with open(file_path_Independent, "rb") as f:
    Independent_Data_dictionary = pickle.load(f)

#Take the useful data
datasets_to_keep = ['fri_c3_1000_50', 'fri_c2_1000_25', 'fri_c4_500_50', 'fri_c4_1000_50', 'fri_c1_1000_25', 'fri_c1_500_50', 'fri_c3_1000_25', 'auto93', 'pyrim', 'autoPrice', 'boston', 'Concrete_Compressive_Strength', 'Auto_MPG', 'Forest Fires', 'Servo', 'Airfoil_Self_Noise', 'Wine_Quality', 'BodyFat', 'California_Housing', 'Quake']

Dependent_Data_dictionary = {
    k: v for k, v in Dependent_Data_dictionary.items()
    if k in datasets_to_keep
}

Independent_Data_dictionary = {
    k: v for k, v in Independent_Data_dictionary.items()
    if k in datasets_to_keep
}

#Write a function to do AFE

def run_afe_nested_cv(
    Independent_Data_dictionary,
    Dependent_Data_dictionary,
    model,
    save_path,
    n_trials=50,
    outer_splits=5,
    inner_splits=5,
):
    """
    Runs 5-fold outer CV + 3-fold inner Optuna HPO for all datasets.

    Parameters
    ----------
    model : sklearn model instance
        Example: LinearRegression(), RandomForestRegressor(), etc.

    save_path : str
        Folder to save dataset pickle files.

    n_trials : int
        Number of Optuna trials.
    """

    os.makedirs(save_path, exist_ok=True)
    outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    
    for dataset_name in Independent_Data_dictionary.keys():

        print(f"\nProcessing {dataset_name}")

        X = Independent_Data_dictionary[dataset_name]
        y = Dependent_Data_dictionary[dataset_name]

        fold_results = {}

        for fold_id, (train_idx, test_idx) in enumerate(outer_kf.split(X), start=1):

            print(f"  Fold {fold_id}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # -------------------------------------------
            # Compute max dcor for this fold
            # -------------------------------------------
            max_dcor = 0
            for col in X_train.columns:
                if col not in X_train.select_dtypes(include=["object", "category"]).columns.tolist():
                    val = distance_correlation(
                        X_train[col].values.astype(float),
                        y_train.values.astype(float)
                    )
                    if val > max_dcor:
                        max_dcor = val
                        
            # -------------------------------------------
            # Inner Optuna
            # -------------------------------------------
            def objective(trial):

                n_int = trial.suggest_int("n_interaction_levels", 1, 5)
                score_bin = trial.suggest_float("score_binary", 0.0, 1.0)
                score_eps = trial.suggest_float("score_eps", 0.0, 1.0)
                dep_th = trial.suggest_float("dep_threshold", 0.0, max_dcor)
                fs_score_mode = trial.suggest_categorical(
                    "fs_score_mode",
                    ['dependency', 'permutation']
                )
                model_aware_fs_mode = trial.suggest_categorical(
                        "model_aware_fs_mode",
                        ['final_only', 'off']
                    )

                wf = AutoFEWorkflow(
                    task="regression",
                    model=model,
                    metric="neg_root_mean_squared_error",
                    iterations=None,
                    k_step=1,
                    score_binary=score_bin,
                    score_eps=score_eps,
                    dep_threshold=dep_th,
                    n_interaction_levels=n_int,
                    random_state=42,
                    model_aware_fs_mode=model_aware_fs_mode,
                    encode_categoricals=True,
                    fs_score_mode=fs_score_mode,
                    categorical_cols=X_train.select_dtypes(include=["object", "category"]).columns.tolist()
                    
                )

                inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
                scores = []

                for tr_idx, val_idx in inner_kf.split(X_train):

                    X_tr = X_train.iloc[tr_idx]
                    y_tr = y_train.iloc[tr_idx]
                    X_val = X_train.iloc[val_idx]
                    y_val = y_train.iloc[val_idx]

                    wf.run(X_tr, y_tr)
                    X_tr_trans = wf.transform_selected(X_tr)
                    X_val_trans = wf.transform_selected(X_val)


                    if X_tr_trans is None or X_tr_trans.shape[1] == 0:
                        raise optuna.exceptions.TrialPruned()
                    

                    model_clone = clone(model)
                    model_clone.fit(X_tr_trans, y_tr)

                    preds = model_clone.predict(X_val_trans)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    scores.append(rmse)

                return np.mean(scores)

            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="minimize",sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            trials_df = study.trials_dataframe()
            best = study.best_params

            # -------------------------------------------
            # Final fit on full fold training data
            # -------------------------------------------
            wf_final = AutoFEWorkflow(
                task="regression",
                model=model,
                metric="neg_root_mean_squared_error",
                iterations=None,
                k_step=1,
                score_binary=best["score_binary"],
                score_eps=best["score_eps"],
                dep_threshold=best["dep_threshold"],
                n_interaction_levels=best["n_interaction_levels"],
                random_state=42,
                model_aware_fs_mode=best['model_aware_fs_mode'],
                encode_categoricals=True,
                fs_score_mode=best["fs_score_mode"],
                categorical_cols=X_train.select_dtypes(include=["object", "category"]).columns.tolist(),
                

            )
            start_time = time.time()
            wf_final.run(X_train, y_train)
            end_time = time.time()
            time_taken = end_time - start_time
            X_train_trans = wf_final.transform_selected(X_train)
            X_test_trans = wf_final.transform_selected(X_test)

            fold_results[f"fold{fold_id}"] = {
                "X_train": X_train,
                "X_train_trans": X_train_trans,
                "X_test": X_test,
                "X_test_trans": X_test_trans,
                "y_train": y_train,
                "y_test": y_test,
                "best_hyperparameters": best,
                "time_taken_seconds": time_taken,
                "optuna_trials": trials_df,   # <-- add this

            }

        with open(os.path.join(save_path, f"{dataset_name}.pkl"), "wb") as f:
            pickle.dump(fold_results, f)

        print(f"Saved {dataset_name}")

#Running the code

save_path = (
    BASE_DIR
    / ".."
    / "Data"
    / "All Data"
    / "Transformed_Data"
    / "RF"
).resolve()

print("Saving to:", save_path)

run_afe_nested_cv(
    Independent_Data_dictionary,
    Dependent_Data_dictionary,
    model=RandomForestRegressor(random_state=42),
    save_path=str(save_path),
    n_trials=50
)
