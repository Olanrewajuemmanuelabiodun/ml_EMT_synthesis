from xgboost import XGBClassifier
#from tabpfn_client import TabPFNClassifier
from tabpfn import TabPFNClassifier
import itertools
import warnings
import yaml
from copy import deepcopy
from joblib import Parallel, delayed

# read data
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import pandas as pd
from json_tricks import dump
from zeolite_emt.data_transformation import (
    feat_dict,
    Transformation,
    PCA_Transformation,
    clean_discrete_columns,
)


def dataset_preprocessing(
    df_in,
    transform_cols,
    input_feat_cols,
    transformation_type="std",
    pca=False,
    pca_dim=None,
):
    transformation = Transformation(df_in, transform_cols)
    if transformation_type == "min_max":
        df = transformation.min_max_transform()  # for min_max_transformation
    else:
        df = transformation.std_transform()  # for standard scale transformation
    pca_object=None
    pca_cols=None
    if pca:
        if pca_dim is None:
            pca_dim = 3
            print("pca_dim (PCA components) not specified, default to 3 for pca=True")
        pca_tr = PCA_Transformation(
            df_in=df, input_cols=input_feat_cols, pca_dim=pca_dim
        )
        df = pca_tr.pca_transform()
        pca_object=pca_tr
        pca_cols=pca_tr.PCA_cols
    return df,{'transformation_type':transformation_type,'transformation_obj':transformation,'pca':pca,'pca_object':pca_object,'pca_cols':pca_cols}


def dataset_split_dict(
    df,
    feat_col,
    label_col,
    seed_list,
    partition=True,
    partition_col=None,
    test_size=1 / 5,
    val_size=1 / 2,
):
    """
    Split a DataFrame into stratified train/val/test sets for one or more seeds.

    If `partition` is True, splits are done separately on 'Region'=='in' and 'Region'=='out'
    subsets and then merged. This preserves label distribution and region representation.

    Args:
        df (pd.DataFrame): Input dataset.
        feat_col (list): List of feature column names.
        label_col (str): Name of the label column.
        seed_list (int or list of int): A single random seed or list of seeds.
        partition (bool): Whether to split based on 'Region' ('in' vs 'out') before stratifying.
        test_size (float): Fraction of full data used for temp set (val + test).
        val_size (float): Fraction of temp set used as test set.

    Returns:
        dict: A dictionary keyed by seed. Each value contains split datasets:
              {
                  'X_train', 'y_train',
                  'X_val', 'y_val',
                  'X_test', 'y_test',
                  ... (plus region-specific splits if partition=True)
              }
    """
    if isinstance(seed_list, int):
        seed_list = [seed_list]

    def stratified_split(data, seed):
        """Performs stratified train/val/test split."""
        train, temp = train_test_split(
            data, test_size=test_size, stratify=data[label_col], random_state=seed
        )
        val, test = train_test_split(
            temp, test_size=val_size, stratify=temp[label_col], random_state=seed
        )
        return train, val, test

    all_splits = {}

    for seed in seed_list:
        if partition:
            partition_col = "Region"
            df_in = df[df[partition_col] == "in"]
            df_out = df[df[partition_col] == "out"]

            train_in, val_in, test_in = stratified_split(df_in, seed)
            train_out, val_out, test_out = stratified_split(df_out, seed)

            split = {}
            for split_name, df1, df2 in zip(
                ["train", "val", "test"],
                [train_in, val_in, test_in],
                [train_out, val_out, test_out],
            ):
                split[f"X_{split_name}_reduced"] = df1[feat_col]
                split[f"y_{split_name}_reduced"] = df1[label_col]
                split[f"X_{split_name}_remaining"] = df2[feat_col]
                split[f"y_{split_name}_remaining"] = df2[label_col]
                split[f"X_{split_name}"] = pd.concat(
                    [df1[feat_col], df2[feat_col]], axis=0
                )
                split[f"y_{split_name}"] = pd.concat(
                    [df1[label_col], df2[label_col]], axis=0
                )

            all_splits[seed] = split
        else:
            train, val, test = stratified_split(df, seed)
            all_splits[seed] = {
                "X_train": train[feat_col],
                "y_train": train[label_col],
                "X_val": val[feat_col],
                "y_val": val[label_col],
                "X_test": test[feat_col],
                "y_test": test[label_col],
            }

    return all_splits


def dataset_split_df(
    df,
    label_col,
    seed_list,
    partition=True,
    partition_col="Region",
    test_size=1 / 5,
    val_size=1 / 2,
):

    if isinstance(seed_list, int):
        seed_list = [seed_list]

    def stratified_split(data, seed):
        train, temp = train_test_split(
            data, test_size=test_size, stratify=data[label_col], random_state=seed
        )
        val, test = train_test_split(
            temp, test_size=val_size, stratify=temp[label_col], random_state=seed
        )
        return train, val, test

    final_outputs = {}

    for seed in seed_list:
        if partition:
            df_in = df[df[partition_col] == "in"]
            df_out = df[df[partition_col] == "out"]

            train_in, val_in, test_in = stratified_split(df_in, seed)
            train_out, val_out, test_out = stratified_split(df_out, seed)

            def annotate_and_concat(df_split, region, split_name):
                df_split = df_split.copy()
                df_split["Region"] = region
                df_split["Split"] = split_name
                return df_split

            frames = []
            for name, df1, df2 in zip(
                ["train", "val", "test"],
                [train_in, val_in, test_in],
                [train_out, val_out, test_out],
            ):
                frames.append(annotate_and_concat(df1, "in", name))
                frames.append(annotate_and_concat(df2, "out", name))

            full_df = pd.concat(frames, ignore_index=True)
        else:
            train, val, test = stratified_split(df, seed)

            def annotate(df_split, split_name):
                df_split = df_split.copy()
                df_split["Split"] = split_name
                return df_split

            full_df = pd.concat(
                [
                    annotate(train, "train"),
                    annotate(val, "val"),
                    annotate(test, "test"),
                ],
                ignore_index=True,
            )

        final_outputs[seed] = full_df

    return final_outputs


class XGB_grid_search:
    def __init__(
        self, param_grid, seed_list, feat_cols,label_col,objective, num_class, eval_metric, average="macro"
    ):
        """
        Initialize the training class with global XGBoost and evaluation settings.
        """
        self.param_grid = param_grid
        self.seed_list = seed_list
        self.objective = objective
        self.num_class = num_class
        self.eval_metric = eval_metric
        self.average = average
        self.feat_cols = feat_cols
        self.label_col = label_col

    def all_params_xgb(self):
        """
        Generate all combinations of hyperparameters from the grid.

        Handles both lists of values (for grid search) and single fixed values
        (which are converted to lists internally).

        Returns:
            List[dict]: List of all hyperparameter combinations as dictionaries.
        """
        # Ensure every value is a list
        normalized_grid = {
            k: v if isinstance(v, list) else [v] for k, v in self.param_grid.items()
        }

        keys = list(normalized_grid.keys())
        values = list(normalized_grid.values())
        all_combinations = list(itertools.product(*values))
        return [dict(zip(keys, v)) for v in all_combinations]


    def evaluate_grid_xgb(self, dict_df, n_jobs=-1, verbose=10):
        """
        Evaluate XGBoost hyperparameters over multiple seeds using DataFrame-based splits.

        Args:
            dict_df (dict): {seed: DataFrame} with all features, 'true_label', 'Split', etc.
            feat_cols (list): Names of feature columns.
            label_col (str): Name of the ground truth label column.
            n_jobs (int): Parallel job count.
            verbose (int): Verbosity level.

        Returns:
            List of dictionaries:
                - 'params': hyperparameter set
                - 'f1_val_array': F1 scores across seeds
                - 'f1_val_mean': mean of those F1 scores
        """

        all_params = self.all_params_xgb()

        def evaluate_single_param(params):
            f1_val_list = []

            for seed, df in dict_df.items():
                df_train = df[df["Split"] == "train"]
                df_val = df[df["Split"] == "val"]

                X_train = df_train[self.feat_cols]
                y_train = df_train[self.label_col]
                X_val = df_val[self.feat_cols]
                y_val = df_val[self.label_col]

                if self.objective in ["binary:logistic", "binary:logitraw"]:
                    xgb_clf = XGBClassifier(
                        objective=self.objective,
                        eval_metric=self.eval_metric,
                        **params,
                    )
                else:
                    xgb_clf = XGBClassifier(
                        objective=self.objective,
                        num_class=self.num_class,
                        eval_metric=self.eval_metric,
                        **params,
                    )

                xgb_clf.fit(X_train, y_train)
                y_val_pred = xgb_clf.predict(X_val)

                f1_val = f1_score(y_val, y_val_pred, average=self.average)
                f1_val_list.append(f1_val)

            mean_f1 = np.mean(f1_val_list)
            print("params:", params, "| mean f1_val:", mean_f1)

            return {
                "params": params,
                "f1_val_array": np.array(f1_val_list),
                "f1_val_mean": mean_f1,
            }

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_single_param)(params) for params in all_params
        )

        return results

    def best_grid_evaluation(self, dict_df):
        """
        Evaluate the best XGBoost hyperparameter set (based on mean F1) across all seeds,
        and return structured per-split metrics with aggregated statistics.

        Args:
            dict_arr (dict): Dictionary mapping seed -> dict of train/val/test splits.

        Returns:
            dict: {
                'best_params': <dict>,
                'train': {
                    'average_stats': {...},
                    'seeds': {
                        <seed>: {...}
                    }
                },
                'val': { ... },
                'test': { ... }
            }
        """

        results = self.evaluate_grid_xgb(dict_df)
        best_result = max(results, key=lambda x: x["f1_val_mean"])
        best_params = best_result["params"]

        print("Best hyperparameters found:")
        print(best_params)
        print(f"Best mean validation F1: {best_result['f1_val_mean']:.4f}")
        return best_params

from copy import deepcopy
from xgboost import XGBClassifier
import numpy as np

class TrainXGB:
    def __init__(self, params, seed_list, feat_cols, label_col, objective, num_class, eval_metric):
        """
        Initializes the training class with XGBoost settings and data configuration.

        Args:
            params (dict): Hyperparameters for XGBoost model.
            seed_list (list): List of random seeds used for dataset splits.
            feat_cols (list): Names of feature columns.
            label_col (str): Name of the target column.
            objective (str): XGBoost objective function.
            num_class (int): Number of classes (for multiclass classification).
            eval_metric (str or list): Evaluation metric(s) used by XGBoost.
        """
        self.params = params
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col
        self.objective = objective
        self.num_class = num_class
        self.eval_metric = eval_metric

    def set_up_XGB(self, dict_df):
        """
        Trains XGBoost models for each seed's dataset and predicts on all rows.

        Uses the 'Split' column in the DataFrame to determine training data.
        Adds columns for predicted labels and predicted probabilities.

        Args:
            dict_df (dict): A dictionary mapping each seed to its corresponding DataFrame.

        Returns:
            dict: Updated dictionary with 'pred_label' and 'proba' columns.
        """
        dict_df_out = deepcopy(dict_df)

        for seed, df in dict_df_out.items():
            df_train = df[df['Split'] == 'train']
            X_train = df_train[self.feat_cols]
            y_train = df_train[self.label_col]

            if self.objective in ["binary:logistic", "binary:logitraw"]:
                xgb_clf = XGBClassifier(
                    objective=self.objective,
                    eval_metric=self.eval_metric,
                    **self.params
                )
            else:
                xgb_clf = XGBClassifier(
                    objective=self.objective,
                    num_class=self.num_class,
                    eval_metric=self.eval_metric,
                    **self.params
                )

            # Fit model
            xgb_clf.fit(X_train, y_train)

            # Predict labels
            df[f"{self.label_col}_pred"] = xgb_clf.predict(df[self.feat_cols])

            # Predict probabilities
            y_proba = xgb_clf.predict_proba(df[self.feat_cols])

            if self.num_class == 2:
                # Binary: save probability for class 1
                df[f"{self.label_col}_proba"] = y_proba[:, 1]
            else:
                # Ternary/multiclass: save full vector of probabilities
                df[f"{self.label_col}_proba"] = y_proba.tolist()

            # Save updated DataFrame
            dict_df_out[seed] = df

        return dict_df_out

# from xgboost import XGBClassifier
# from copy import deepcopy

# class TrainXGB:
#     def __init__(self, params, seed_list, feat_cols, label_col, objective, num_class, eval_metric):
#         """
#         Initializes the training class with XGBoost settings and data configuration.
#         """
#         self.params = params
#         self.seed_list = seed_list
#         self.feat_cols = feat_cols
#         self.label_col = label_col
#         self.objective = objective
#         self.num_class = num_class
#         self.eval_metric = eval_metric

#     def set_up_XGB(self, dict_df):
#         """
#         Trains XGBoost models for each seed's dataset and predicts on all rows.
#         Adds 'pred' and 'proba' columns to each DataFrame.
#         """
#         dict_df_out = deepcopy(dict_df)

#         for seed, df in dict_df_out.items():
#             df_train = df[df['Split'] == 'train']
#             X_train = df_train[self.feat_cols]
#             y_train = df_train[self.label_col]

#             xgb_clf = XGBClassifier(
#                 objective=self.objective,
#                 eval_metric=self.eval_metric,
#                 num_class=self.num_class if self.num_class > 2 else None,
#                 use_label_encoder=False,
#                 **self.params
#             )

#             xgb_clf.fit(X_train, y_train)

#             # Save predicted labels and probabilities
#             df[f"{self.label_col}_pred"] = xgb_clf.predict(df[self.feat_cols])
#             df[f"{self.label_col}_proba"] = xgb_clf.predict_proba(df[self.feat_cols]).tolist()

#             dict_df_out[seed] = df

#         return dict_df_out


class TrainTabPFN:
    def __init__(self, seed_list, feat_cols, label_col):
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col

    def set_up_TabPFN(self, dict_df):
        """
        Train TabPFN per-seed on the train split and add:
           <label_col>_pred   (predicted class)
           <label_col>_proba  (float list for binary, list-of-lists for multiclass)
        """
        dict_df_out = deepcopy(dict_df)

        for seed, df in dict_df_out.items():
            df_train = df[df["Split"] == "train"]
            X_train = df_train[self.feat_cols]
            y_train = df_train[self.label_col]

            clf = TabPFNClassifier()  # TabPFN has no real HPs to tune
            clf.fit(X_train, y_train)

            # Predictions on ALL rows
            X_all = df[self.feat_cols]
            y_pred = clf.predict(X_all)
            y_proba = clf.predict_proba(X_all)

            df[f"{self.label_col}_pred"] = y_pred

            if y_proba.shape[1] == 2:  # binary -> store prob of class 1
                df[f"{self.label_col}_proba"] = y_proba[:, 1].tolist()
            else:                       # multiclass -> store full vector
                df[f"{self.label_col}_proba"] = y_proba.tolist()

            dict_df_out[seed] = df

        return dict_df_out


# class TrainTabPFN:
#     def __init__(self, seed_list, feat_cols, label_col):
#         """
#         Initializes the training class with XGBoost settings and data configuration.

#         Args:
#             seed_list (list): List of random seeds used for dataset splits.
#             feat_cols (list): Names of feature columns.
#             label_col (str): Name of the target column.
#         """
#         self.seed_list = seed_list
#         self.feat_cols = feat_cols
#         self.label_col = label_col

#     def set_up_TabPFN(self, dict_df):
#         """
#         Trains TabPFN models for each seed's dataset and predicts on all rows.

#         Uses the 'Split' column in the DataFrame to determine training data.
#         Adds a new 'pred_label' column with model predictions.

#         Args:
#             dict_df (dict): A dictionary mapping each seed to its corresponding DataFrame.
#                             Each DataFrame must include 'Split', feature columns, and the true label column.

#         Returns:
#             dict: Updated dictionary with 'pred_label' column added to each seed's DataFrame.
#         """
#         dict_df_out = deepcopy(dict_df)

#         for seed, df in dict_df_out.items():
#             df_train = df[df['Split'] == 'train']
#             X_train = df_train[self.feat_cols]
#             y_train = df_train[self.label_col]

#             tabpfn_clf = TabPFNClassifier()

#             # Train and predict on full dataset
#             tabpfn_clf.fit(X_train, y_train)
#             df[f"{self.label_col}_pred"] = tabpfn_clf.predict(df[self.feat_cols])

#             # Save updated DataFrame
#             dict_df_out[seed] = df

#         return dict_df_out


def compute_seed_metrics(dict_df_out, label_col, pred_col,average='macro'):
    """
    Compute classification metrics across seeds and splits.

    Args:
        dict_df_out (dict): {seed: DataFrame with Split, Region, true/pred labels}
        label_col (str): Name of true label column.
        pred_col (str): Name of predicted label column.

    Returns:
        dict: Dictionary with per-seed and summary statistics.
    """
    metrics = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    splits = ['train', 'val', 'test']
    per_seed_stats = {}
    summary_stats = {m: {s: [] for s in splits} for m in metrics}

    for seed, df in dict_df_out.items():
        per_seed_stats[seed] = {}
        for metric_name, metric_func in metrics.items():
            per_seed_stats[seed][metric_name] = {}
            for split in splits:
                split_df = df[df['Split'] == split]
                if split_df.empty:
                    score = np.nan
                else:
                    if metric_name == 'accuracy':
                        score = metric_func(
                            split_df[label_col],
                            split_df[pred_col]
                        )
                    else:
                        score = metric_func(
                            split_df[label_col],
                            split_df[pred_col],
                            average=average
                        )
                per_seed_stats[seed][metric_name][split] = score
                summary_stats[metric_name][split].append(score)

    # Compute mean and std for each metric/split
    summary = {}
    for metric_name in metrics:
        summary[metric_name] = {}
        for split in splits:
            values = summary_stats[metric_name][split]
            values = [v for v in values if not np.isnan(v)]
            summary[metric_name][split] = {
                'mean': float(np.mean(values)) if values else None,
                'std': float(np.std(values)) if values else None,
            }

    return {'seeds': per_seed_stats, 'summary': summary}


# from sklearn.ensemble import RandomForestClassifier


# class RF_grid_search:
#     def __init__(
#         self, param_grid, seed_list, feat_cols,label_col,objective, num_class, eval_metric, average="macro"
#     ):
#         """
#         Initialize the training class with global RF and evaluation settings.
#         """
#         self.param_grid = param_grid
#         self.seed_list = seed_list
#         self.objective = objective
#         self.num_class = num_class
#         self.eval_metric = eval_metric
#         self.average = average
#         self.feat_cols = feat_cols
#         self.label_col = label_col

#     def all_params_rf(self):
#         """
#         Generate all combinations of hyperparameters from the grid.

#         Handles both lists of values (for grid search) and single fixed values
#         (which are converted to lists internally).

#         Returns:
#             List[dict]: List of all hyperparameter combinations as dictionaries.
#         """
#         # Ensure every value is a list
#         normalized_grid = {
#             k: v if isinstance(v, list) else [v] for k, v in self.param_grid.items()
#         }

#         keys = list(normalized_grid.keys())
#         values = list(normalized_grid.values())
#         all_combinations = list(itertools.product(*values))
#         return [dict(zip(keys, v)) for v in all_combinations]


#     def evaluate_grid_rf(self, dict_df, n_jobs=-1, verbose=10):
#         """
#         Evaluate RF hyperparameters over multiple seeds using DataFrame-based splits.

#         Args:
#             dict_df (dict): {seed: DataFrame} with all features, 'true_label', 'Split', etc.
#             feat_cols (list): Names of feature columns.
#             label_col (str): Name of the ground truth label column.
#             n_jobs (int): Parallel job count.
#             verbose (int): Verbosity level.

#         Returns:
#             List of dictionaries:
#                 - 'params': hyperparameter set
#                 - 'f1_val_array': F1 scores across seeds
#                 - 'f1_val_mean': mean of those F1 scores
#         """

#         all_params = self.all_params_rf()

#         def evaluate_single_param(params):
#             f1_val_list = []

#             for seed, df in dict_df.items():
#                 df_train = df[df["Split"] == "train"]
#                 df_val = df[df["Split"] == "val"]

#                 X_train = df_train[self.feat_cols]
#                 y_train = df_train[self.label_col]
#                 X_val = df_val[self.feat_cols]
#                 y_val = df_val[self.label_col]
                
#                 rf_clf = RandomForestClassifier(**params,random_state=seed)


#                 rf_clf.fit(X_train, y_train)
#                 y_val_pred = rf_clf.predict(X_val)

#                 f1_val = f1_score(y_val, y_val_pred, average=self.average)
#                 f1_val_list.append(f1_val)

#             mean_f1 = np.mean(f1_val_list)
#             print("params:", params, "| mean f1_val:", mean_f1)

#             return {
#                 "params": params,
#                 "f1_val_array": np.array(f1_val_list),
#                 "f1_val_mean": mean_f1,
#             }

#         results = Parallel(n_jobs=1, verbose=verbose)(
#             delayed(evaluate_single_param)(params) for params in all_params
#         )

#         return results

#     def best_grid_evaluation(self, dict_df):
#         """
#         Evaluate the best RF hyperparameter set (based on mean F1) across all seeds,
#         and return structured per-split metrics with aggregated statistics.

#         Args:
#             dict_arr (dict): Dictionary mapping seed -> dict of train/val/test splits.

#         Returns:
#             dict: {
#                 'best_params': <dict>,
#                 'train': {
#                     'average_stats': {...},
#                     'seeds': {
#                         <seed>: {...}
#                     }
#                 },
#                 'val': { ... },
#                 'test': { ... }
#             }
#         """

#         results = self.evaluate_grid_rf(dict_df)
#         best_result = max(results, key=lambda x: x["f1_val_mean"])
#         best_params = best_result["params"]

#         print("Best hyperparameters found:")
#         print(best_params)
#         print(f"Best mean validation F1: {best_result['f1_val_mean']:.4f}")
#         return best_params




# class TrainRF:
#     def __init__(self, params, seed_list, feat_cols, label_col, objective, num_class, eval_metric):
#         """
#         Initializes the training class with RF settings and data configuration.

#         Args:
#             params (dict): Hyperparameters for XGBoost model.
#             seed_list (list): List of random seeds used for dataset splits.
#             feat_cols (list): Names of feature columns.
#             label_col (str): Name of the target column.
#             objective (str): XGBoost objective function.
#             num_class (int): Number of classes (for multiclass classification).
#             eval_metric (str or list): Evaluation metric(s) used by XGBoost.
#         """
#         self.params = params
#         self.seed_list = seed_list
#         self.feat_cols = feat_cols
#         self.label_col = label_col
#         self.objective = objective
#         self.num_class = num_class
#         self.eval_metric = eval_metric

#     def set_up_RF(self, dict_df):
#         """
#         Trains RF models for each seed's dataset and predicts on all rows.

#         Uses the 'Split' column in the DataFrame to determine training data.
#         Adds a new 'pred_label' column with model predictions.

#         Args:
#             dict_df (dict): A dictionary mapping each seed to its corresponding DataFrame.
#                             Each DataFrame must include 'Split', feature columns, and the true label column.

#         Returns:
#             dict: Updated dictionary with 'pred_label' column added to each seed's DataFrame.
#         """
#         dict_df_out = deepcopy(dict_df)

#         for seed, df in dict_df_out.items():
#             df_train = df[df['Split'] == 'train']
#             X_train = df_train[self.feat_cols]
#             y_train = df_train[self.label_col]

#             rf_clf = RandomForestClassifier(**self.params,random_state=seed)


#             # Train and predict on full dataset
#             rf_clf.fit(X_train, y_train)
#             df[f"{self.label_col}_pred"] = rf_clf.predict(df[self.feat_cols])

#             # Save updated DataFrame
#             dict_df_out[seed] = df

#         return dict_df_out




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np
import itertools

class RF_grid_search:
    def __init__(self, param_grid, seed_list, feat_cols, label_col, average="macro"):
        self.param_grid = param_grid
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col
        self.average = average

    def all_params_rf(self):
        normalized_grid = {k: v if isinstance(v, list) else [v] for k, v in self.param_grid.items()}
        keys = list(normalized_grid.keys())
        values = list(normalized_grid.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def evaluate_grid_rf(self, dict_df, n_jobs=-1, verbose=10):
        all_params = self.all_params_rf()

        def evaluate_single_param(params):
            f1_val_list = []
            for seed, df in dict_df.items():
                df_train = df[df["Split"] == "train"]
                df_val = df[df["Split"] == "val"]
                X_train, y_train = df_train[self.feat_cols], df_train[self.label_col]
                X_val, y_val = df_val[self.feat_cols], df_val[self.label_col]

                clf = RandomForestClassifier(**params, random_state=seed)
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                f1_val = f1_score(y_val, y_val_pred, average=self.average)
                f1_val_list.append(f1_val)

            return {
                "params": params,
                "f1_val_array": np.array(f1_val_list),
                "f1_val_mean": np.mean(f1_val_list)
            }

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_single_param)(params) for params in all_params
        )
        return results

    def best_grid_evaluation(self, dict_df):
        results = self.evaluate_grid_rf(dict_df)
        best_result = max(results, key=lambda x: x["f1_val_mean"])
        print("Best RF hyperparameters:", best_result["params"])
        return best_result["params"]

class TrainRF:
    def __init__(self, params, seed_list, feat_cols, label_col):
        self.params = params
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col

    def set_up_RF(self, dict_df):
        dict_df_out = deepcopy(dict_df)

        for seed, df in dict_df_out.items():
            df_train = df[df["Split"] == "train"]
            X_train = df_train[self.feat_cols]
            y_train = df_train[self.label_col]

            clf = RandomForestClassifier(**self.params, random_state=seed)
            clf.fit(X_train, y_train)

            # Predict on all rows
            df[f"{self.label_col}_pred"] = clf.predict(df[self.feat_cols])

            proba = clf.predict_proba(df[self.feat_cols])
            if proba.shape[1] == 2:
                # Binary classification: use class 1 probability
                df[f"{self.label_col}_proba"] = proba[:, 1].tolist()
            else:
                # Multiclass: store full array
                df[f"{self.label_col}_proba"] = proba.tolist()

            dict_df_out[seed] = df

        return dict_df_out
#here
def print_class_distributions(dict_df, label_col, split="val"):
    """
    Print class distributions for a specific split ('train', 'val', or 'test') in each seed's dataset.

    Args:
        dict_df (dict): Dictionary mapping seed to its corresponding DataFrame.
        label_col (str): Name of the label column.
        split (str): Which split to check ('train', 'val', 'test').
    """
    print(f"\n🔍 Checking class distribution in {split} sets (before training):")
    for seed, df in dict_df.items():
        if split in df["Split"].unique():
            class_counts = df[df["Split"] == split][label_col].value_counts()
            print(f"Seed {seed} - {split} set class distribution:\n{class_counts.to_string()}\n")
        else:
            print(f"Seed {seed} - Split '{split}' not found.\n")



def print_class_distributions(dict_df, label_col, split="val", expected_classes=None):
    """
    Print class distributions for a specific split ('train', 'val', or 'test') in each seed's dataset.
    Also warns if any expected classes are missing in the split.

    Args:
        dict_df (dict): Dictionary mapping seed to its corresponding DataFrame.
        label_col (str): Name of the label column.
        split (str): Which split to check ('train', 'val', 'test').
        expected_classes (list or None): List of expected class labels. If provided, missing classes will be reported.
    """
    for seed, df in dict_df.items():
        if split in df["Split"].unique():
            df_split = df[df["Split"] == split]
            class_counts = df_split[label_col].value_counts().sort_index()
            print(f"Seed {seed} - {split} set class distribution:\n{class_counts.to_string()}")

            if expected_classes is not None:
                present_classes = set(class_counts.index.tolist())
                missing = set(expected_classes) - present_classes
                if missing:
                    print(f"⚠️  Seed {seed} is missing classes in '{split}': {sorted(missing)}")
            print()  # extra newline for readability
        else:
            print(f"Seed {seed} - Split '{split}' not found.\n")



from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np
import itertools

class CatBoostGridSearch:
    def __init__(self, param_grid, seed_list, feat_cols, label_col, average="macro"):
        self.param_grid = param_grid
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col
        self.average = average

    def all_params_catboost(self):
        normalized_grid = {k: v if isinstance(v, list) else [v] for k, v in self.param_grid.items()}
        keys = list(normalized_grid.keys())
        values = list(normalized_grid.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def evaluate_grid_catboost(self, dict_df, n_jobs=-1, verbose=10):
        all_params = self.all_params_catboost()

        def evaluate_single_param(params):
            f1_val_list = []
            for seed, df in dict_df.items():
                df_train = df[df["Split"] == "train"]
                df_val = df[df["Split"] == "val"]
                X_train, y_train = df_train[self.feat_cols], df_train[self.label_col]
                X_val, y_val = df_val[self.feat_cols], df_val[self.label_col]

                clf = CatBoostClassifier(**params, random_seed=seed, verbose=0)
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                f1_val = f1_score(y_val, y_val_pred, average=self.average)
                f1_val_list.append(f1_val)

            return {
                "params": params,
                "f1_val_array": np.array(f1_val_list),
                "f1_val_mean": np.mean(f1_val_list)
            }

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_single_param)(params) for params in all_params
        )
        return results

    def best_grid_evaluation(self, dict_df):
        results = self.evaluate_grid_catboost(dict_df)
        best_result = max(results, key=lambda x: x["f1_val_mean"])
        print("Best CatBoost hyperparameters:", best_result["params"])
        return best_result["params"]


class TrainCatBoost:
    def __init__(self, params, seed_list, feat_cols, label_col):
        self.params = params
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col

    def set_up_catboost(self, dict_df):
        dict_df_out = deepcopy(dict_df)

        for seed, df in dict_df_out.items():
            df_train = df[df["Split"] == "train"]
            X_train = df_train[self.feat_cols]
            y_train = df_train[self.label_col]

            clf = CatBoostClassifier(**self.params, random_seed=seed, verbose=0)
            clf.fit(X_train, y_train)

            # Predict on all rows
            df[f"{self.label_col}_pred"] = clf.predict(df[self.feat_cols])

            proba = clf.predict_proba(df[self.feat_cols])
            if proba.shape[1] == 2:
                df[f"{self.label_col}_proba"] = proba[:, 1].tolist()
            else:
                df[f"{self.label_col}_proba"] = proba.tolist()

            dict_df_out[seed] = df

        return dict_df_out


from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np
import itertools

class LightGBMGridSearch:
    def __init__(self, param_grid, seed_list, feat_cols, label_col, average="macro"):
        self.param_grid = param_grid
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col
        self.average = average

    def all_params_lgbm(self):
        normalized_grid = {k: v if isinstance(v, list) else [v] for k, v in self.param_grid.items()}
        keys = list(normalized_grid.keys())
        values = list(normalized_grid.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def evaluate_grid_lgbm(self, dict_df, n_jobs=-1, verbose=10):
        all_params = self.all_params_lgbm()

        def evaluate_single_param(params):
            f1_val_list = []
            for seed, df in dict_df.items():
                df_train = df[df["Split"] == "train"]
                df_val = df[df["Split"] == "val"]
                X_train, y_train = df_train[self.feat_cols], df_train[self.label_col]
                X_val, y_val = df_val[self.feat_cols], df_val[self.label_col]

                clf = LGBMClassifier(**params, random_state=seed)
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                f1_val = f1_score(y_val, y_val_pred, average=self.average)
                f1_val_list.append(f1_val)

            return {
                "params": params,
                "f1_val_array": np.array(f1_val_list),
                "f1_val_mean": np.mean(f1_val_list)
            }

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_single_param)(params) for params in all_params
        )
        return results

    def best_grid_evaluation(self, dict_df):
        results = self.evaluate_grid_lgbm(dict_df)
        best_result = max(results, key=lambda x: x["f1_val_mean"])
        print("Best LightGBM hyperparameters:", best_result["params"])
        return best_result["params"]


def sanitize_columns(columns):
    return [col.replace(':', '_')
                .replace('/', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('²', '2')
                .replace('₃', '3')
                .replace(' ', '_')
            for col in columns]


class TrainLightGBM:
    def __init__(self, params, seed_list, feat_cols, label_col):
        self.params = params
        self.seed_list = seed_list
        self.feat_cols = feat_cols
        self.label_col = label_col

    def set_up_lightgbm(self, dict_df):
        dict_df_out = deepcopy(dict_df)

        sanitized_feat_cols = sanitize_columns(self.feat_cols)

        for seed, df in dict_df_out.items():
            df.columns = sanitize_columns(df.columns)

            df_train = df[df["Split"] == "train"]
            X_train = df_train[sanitized_feat_cols]
            y_train = df_train[self.label_col]

            clf = LGBMClassifier(**self.params, random_state=seed)
            print(f"Seed {seed} - y_train counts:\n{y_train.value_counts()}")
            print(f"Feature variance:\n{X_train.nunique()}")

            clf.fit(X_train, y_train, callbacks=[log_evaluation(1)])



            # Predict on all rows
            df[f"{self.label_col}_pred"] = clf.predict(df[sanitized_feat_cols])

            proba = clf.predict_proba(df[sanitized_feat_cols])
            if proba.shape[1] == 2:
                df[f"{self.label_col}_proba"] = proba[:, 1].tolist()
            else:
                df[f"{self.label_col}_proba"] = proba.tolist()

            dict_df_out[seed] = df

        return dict_df_out
