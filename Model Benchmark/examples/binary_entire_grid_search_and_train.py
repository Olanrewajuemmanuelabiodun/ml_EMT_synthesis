# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# import yaml
# import os

# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from zeolite_emt.data_transformation import (
#     feat_dict,
#     Transformation,
#     PCA_Transformation,
#     clean_discrete_columns,
#     dict_to_json,
#     json_to_dict,
#     filter_dict_by_region,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     XGB_grid_search,
#     TrainXGB,compute_seed_metrics,
# )

# with open("params.yaml", "r") as file:
#     config = yaml.safe_load(file)


# param_grid = config["param_grid_xgb"]
# ternary_params = config["ternary_classification_parameters"]
# #binary_params = config["binary_classification_parameters"]


# processed_file = os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv')
# df_main=pd.read_csv(processed_file)
# df_main["label_ter"] = df_main["label_ter"].replace(
#     {"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0}
# )
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})
# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols = feat_dict["discrete_feat_cols"]
# input_feat_cols = continuous_feat_cols + discrete_feat_cols


# param_dict = ternary_params

# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [494,101,283,471,326]]

# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [77,273,486,490,101,494,471,283,336,28,326,155,86,17,281,136]]

# # random_seed = 35
# # np.random.seed(random_seed)
# # seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()



# ##########################

# # In the following we split the dataset. partition is True will inforce stratified partition
# # across 'in' region (Na, Si/Al condition ) and 'out' region followed by final lable. False will just
# # make stratifed split based on the final label

# ##########################
# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=False,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )



# ###############################

# # Below we do the preprocessing steps (standard/minmax transform, PCA etc.) for moving forward

# # For considering entire range, use below (by default the region is entire)
# df_forward = filter_dict_by_region(df_process,region=None)

# # For considering only the reduced range, use below
# #df_forward = filter_dict_by_region(df_process,region='in')
# #print(df_forward[77].shape)

# #We further record the transformed dataset and the necessary transformation for inverse transformation later.
# #################################


# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )



# xgb_grid_search = XGB_grid_search(
#     param_grid=param_grid,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     objective=param_dict["objective"],
#     num_class=param_dict["num_class"],
#     eval_metric=param_dict["eval_metric"],
#     average=param_dict["average"],
# )
# best_params_dict = xgb_grid_search.best_grid_evaluation(dict_df=processed_df)

# """
# best_params_dict = {
#     "n_estimators": 8,
#     "max_depth": 4,
#     "alpha": 1,
#     "gamma": 0.0,
#     "learning_rate": 1.0,
#     "subsample": 1.0,
#     "colsample_bytree": 1.0
# }
# """
# print(best_params_dict)

# train_xgb = TrainXGB(
#     params=best_params_dict,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     objective=param_dict["objective"],
#     num_class=param_dict["num_class"],
#     eval_metric=param_dict["eval_metric"],
# )

# dict_df_out = train_xgb.set_up_XGB(processed_df)

# dict_df_out_forward = filter_dict_by_region(dict_df_out,region=None)
# #print(dict_df_out_forward[77].shape)
# dict_summary=compute_seed_metrics(dict_df_out=dict_df_out_forward,label_col=param_dict["label_col"],pred_col=f"{param_dict['label_col']}_pred",average='macro')
# print(dict_summary['summary'])

# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": best_params_dict,
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics":dict_summary,
# }
# dict_to_json(final_dict, "5.json")





#rf

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# import warnings
# warnings.simplefilter('always')


# import yaml
# import os

# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from zeolite_emt.data_transformation import (
#     feat_dict,
#     Transformation,
#     PCA_Transformation,
#     clean_discrete_columns,
#     dict_to_json,
#     json_to_dict,
#     filter_dict_by_region,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     RF_grid_search,
#     TrainRF,compute_seed_metrics,
# )

# with open("examples/params.yaml", "r") as file:
#     config = yaml.safe_load(file)


# param_grid = config["param_grid_rf"]
# ternary_params = config["ternary_classification_parameters"]
# binary_params = config["binary_classification_parameters"]


# processed_file = os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv')
# df_main=pd.read_csv(processed_file)
# df_main["label_ter"] = df_main["label_ter"].replace(
#     {"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0}
# )
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})
# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols = feat_dict["discrete_feat_cols"]
# input_feat_cols = continuous_feat_cols + discrete_feat_cols


# param_dict = binary_params
# random_seed = 35
# np.random.seed(random_seed)
# seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()



# ##########################

# # In the following we split the dataset. partition is True will inforce stratified partition
# # across 'in' region (Na, Si/Al condition ) and 'out' region followed by final lable. False will just
# # make stratifed split based on the final label

# ##########################
# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=False,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )



# ###############################

# # Below we do the preprocessing steps (standard/minmax transform, PCA etc.) for moving forward

# # For considering entire range, use below (by default the region is entire)
# df_forward = filter_dict_by_region(df_process,region=None)

# # For considering only the reduced range, use below
# #df_forward = filter_dict_by_region(df_process,region='in')
# print(df_forward[77].shape)

# #We further record the transformed dataset and the necessary transformation for inverse transformation later.
# #################################


# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )



# rf_grid_search = RF_grid_search(
#     param_grid=param_grid,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=binary_params["label_col"],
#     objective=param_dict["objective"],
#     num_class=param_dict["num_class"],
#     eval_metric=param_dict["eval_metric"],
#     average=param_dict["average"],
# )
# best_params_dict = rf_grid_search.best_grid_evaluation(dict_df=processed_df)

# """
# best_params_dict = {
#     "n_estimators": 8,
#     "max_depth": 4,
#     "alpha": 1,
#     "gamma": 0.0,
#     "learning_rate": 1.0,
#     "subsample": 1.0,
#     "colsample_bytree": 1.0
# }
# """
# print(best_params_dict)

# train_rf = TrainRF(
#     params=best_params_dict,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     objective=param_dict["objective"],
#     num_class=param_dict["num_class"],
#     eval_metric=param_dict["eval_metric"],
# )

# dict_df_out = train_rf.set_up_RF(processed_df)

# dict_df_out_forward = filter_dict_by_region(dict_df_out,region='in')
# print(dict_df_out_forward[77].shape)
# dict_summary=compute_seed_metrics(dict_df_out=dict_df_out_forward,label_col=param_dict["label_col"],pred_col=f"{param_dict['label_col']}_pred",average='macro')
# print(dict_summary['summary'])

# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": best_params_dict,
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics":dict_summary,
# }
# dict_to_json(final_dict, "test_binary_reduced_json.json")

























#randomforest
#Updated Random Forest pipeline with interpolation
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import yaml
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.preprocessing import label_binarize

# from zeolite_emt.data_transformation import (
#     feat_dict,
#     Transformation,
#     PCA_Transformation,
#     clean_discrete_columns,
#     dict_to_json,
#     json_to_dict,
#     filter_dict_by_region,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     RF_grid_search,
#     TrainRF,
#     compute_seed_metrics,
#     print_class_distributions,
# )

# # ===== Load config =====
# with open("params.yaml", "r") as file:
#     config = yaml.safe_load(file)

# # ===== Task Type =====
# task_type = "binary"  # or "binary"
# param_dict = config["binary_classification_parameters"] if task_type == "binary" else config["ternary_classification_parameters"]
# param_grid_rf = config["param_grid_rf"]

# # ===== Load and preprocess dataset =====
# df_main = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv'))
# df_main["label_ter"] = df_main["label_ter"].replace({"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0})
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})

# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols = feat_dict["discrete_feat_cols"]
# input_feat_cols = continuous_feat_cols + discrete_feat_cols


# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [100,409,494,285,117]]


# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [77,273,486,490,101,494,471,283,336,28,326,155,86,17,281,136]]
# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [480, 155, 454, 176,292,163,205,315,136,474,101,212,80,77,117,471]]
# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [454,224,130,409,17,22,62,163,371,315,474,101,490,6,273,28]]
# # ===== Seeds and splitting =====
# # np.random.seed(35)
# # seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()

# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=True,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )

# df_forward = filter_dict_by_region(df_process, region=None)
# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )

# # ===== Hyperparameter search =====
# rf_grid_search = RF_grid_search(
#     param_grid=param_grid_rf,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     average=param_dict["average"],
# )
# best_params_dict = rf_grid_search.best_grid_evaluation(dict_df=processed_df)
# print("Best Parameters:", best_params_dict)

# # ===== Train model =====
# train_rf = TrainRF(
#     params=best_params_dict,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
# )
# dict_df_out = train_rf.set_up_RF(processed_df)

# # ===== ROC averaging =====
# n_classes = param_dict["num_class"]
# classes = list(range(n_classes))
# split_fpr_fixed = np.linspace(0, 1, 101)

# split_tpr_sum = {split: {cls: np.zeros_like(split_fpr_fixed) for cls in classes} for split in ['train', 'val', 'test']}
# split_count = {split: {cls: 0 for cls in classes} for split in ['train', 'val', 'test']}
# auc_results = {}
# split_auc_lists = {"train": [], "val": [], "test": []}
# total_interp_points = 0

# for seed in seed_list:
#     df = dict_df_out[seed]
#     auc_results[seed] = {}

#     for split in ['train', 'val', 'test']:
#         df_split = df[df["Split"] == split]
#         if df_split.empty:
#             continue

#         y_true = df_split[param_dict["label_col"]]
#         y_proba = np.array(df_split[f"{param_dict['label_col']}_proba"].tolist())

#         if n_classes == 2:
#             auc = roc_auc_score(y_true, y_proba)
#             auc_results[seed][split] = {"auc": float(auc)}
#             split_auc_lists[split].append(auc)

#             fpr, tpr, _ = roc_curve(y_true, y_proba)
#             tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#             split_tpr_sum[split][0] += tpr_interp
#             split_count[split][0] += 1
#             total_interp_points += len(split_fpr_fixed)

#         else:
#             y_true_bin = label_binarize(y_true, classes=classes)
#             auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
#             auc_results[seed][split] = {"macro_auc": float(auc_macro)}
#             split_auc_lists[split].append(auc_macro)

#             for i in classes:
#                 try:
#                     fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
#                     tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#                     split_tpr_sum[split][i] += tpr_interp
#                     split_count[split][i] += 1
#                     total_interp_points += len(split_fpr_fixed)
#                 except ValueError:
#                     continue

# # ===== Plot averaged ROC curves in one figure =====
# plt.figure(figsize=(8, 6))

# for split in ['train', 'val', 'test']:
#     for i in classes:
#         if split_count[split][i] > 0:
#             avg_tpr = split_tpr_sum[split][i] / split_count[split][i]
#             label = f'{split.capitalize()} - Class {i}' if n_classes > 2 else f'{split.capitalize()}'
#             plt.plot(split_fpr_fixed, avg_tpr, label=label, linewidth=2)

# # Axis labeling and styling
# plt.xlabel("False Positive Rate", fontsize=20)
# plt.ylabel("True Positive Rate", fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(True)

# # Shift x-axis margin slightly to right
# plt.xlim([-0.03, 1.05])  # 👈 Small negative x limit for margin
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(f"b_rf_{task_type}.png")
# plt.close()


# print(f"\n✅ Total interpolated points across all seeds and splits: {total_interp_points}")

# # ===== Average AUC scores =====
# average_auc_scores = {
#     split: float(np.mean(scores)) if scores else None
#     for split, scores in split_auc_lists.items()
# }

# # ===== Compute final metrics =====
# dict_df_out_forward = filter_dict_by_region(dict_df_out, region=None)
# dict_summary = compute_seed_metrics(
#     dict_df_out=dict_df_out_forward,
#     label_col=param_dict["label_col"],
#     pred_col=f"{param_dict['label_col']}_pred",
#     average='macro'
# )

# # ===== Save results =====
# output_file = f"b_{task_type}_rf.json"
# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": best_params_dict,
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics": dict_summary,
#     "auc_scores": auc_results,
#     "average_auc_scores": average_auc_scores,
# }
# dict_to_json(final_dict, output_file)

# print_class_distributions(processed_df, label_col=param_dict["label_col"], split="val", expected_classes=classes)


# catboost

# catboost_pipeline.py
# catboost_pipeline.py
# catboost_pipeline.py
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import yaml
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.preprocessing import label_binarize

# from zeolite_emt.data_transformation import (
#     feat_dict,
#     Transformation,
#     PCA_Transformation,
#     clean_discrete_columns,
#     dict_to_json,
#     json_to_dict,
#     filter_dict_by_region,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     CatBoostGridSearch,
#     TrainCatBoost,
#     compute_seed_metrics,
#     print_class_distributions,
# )

# # === Load Config ===
# with open("params.yaml", "r") as file:
#     config = yaml.safe_load(file)

# # === Task Type ===
# task_type = "binary"  # or "binary"
# param_dict = config["binary_classification_parameters"] if task_type == "binary" else config["ternary_classification_parameters"]
# param_grid_cb = config["param_grid_catboost"]

# # === Load Dataset ===
# df_main = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv'))
# df_main["label_ter"] = df_main["label_ter"].replace({"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0})
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})

# # === Feature Columns ===
# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols = feat_dict["discrete_feat_cols"]
# input_feat_cols = continuous_feat_cols + discrete_feat_cols

# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [28,494,263,100,326]]

# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [77,273,486,490,101,494,471,283,336,28,326,155,86,17,281,136]]
# # np.random.seed(35)
# # seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
# #              if s not in [480, 155, 454, 176,292,163,205,315,136,474,101,212,80,77,117,471]]
# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [454,224,130,409,17,22,62,163,371,315,474,101,490,6,273,28]]
# # === Seeds ===
# # np.random.seed(35)
# # seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()

# # === Dataset Split ===
# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=True,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )

# df_forward = filter_dict_by_region(df_process, region=None)
# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )

# # === Grid Search ===
# cb_grid_search = CatBoostGridSearch(
#     param_grid=param_grid_cb,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     average=param_dict["average"],
# )
# best_params_dict = cb_grid_search.best_grid_evaluation(dict_df=processed_df)
# print("Best Parameters:", best_params_dict)

# # === Train CatBoost ===
# train_cb = TrainCatBoost(
#     params=best_params_dict,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
# )
# dict_df_out = train_cb.set_up_catboost(processed_df)

# # === ROC Averaging and Plotting ===
# n_classes = param_dict["num_class"]
# classes = list(range(n_classes))
# split_fpr_fixed = np.linspace(0, 1, 101)

# split_tpr_sum = {split: {cls: np.zeros_like(split_fpr_fixed) for cls in classes} for split in ['train', 'val', 'test']}
# split_count = {split: {cls: 0 for cls in classes} for split in ['train', 'val', 'test']}
# auc_results = {}
# split_auc_lists = {"train": [], "val": [], "test": []}
# total_interp_points = 0

# for seed in seed_list:
#     df = dict_df_out[seed]
#     auc_results[seed] = {}

#     for split in ['train', 'val', 'test']:
#         df_split = df[df["Split"] == split]
#         if df_split.empty:
#             continue

#         y_true = df_split[param_dict["label_col"]]
#         y_proba = np.array(df_split[f"{param_dict['label_col']}_proba"].tolist())

#         if n_classes == 2:
#             auc = roc_auc_score(y_true, y_proba)
#             auc_results[seed][split] = {"auc": float(auc)}
#             split_auc_lists[split].append(auc)

#             fpr, tpr, _ = roc_curve(y_true, y_proba)
#             tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#             split_tpr_sum[split][0] += tpr_interp
#             split_count[split][0] += 1
#             total_interp_points += len(split_fpr_fixed)

#         else:
#             y_true_bin = label_binarize(y_true, classes=classes)
#             auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
#             auc_results[seed][split] = {"macro_auc": float(auc_macro)}
#             split_auc_lists[split].append(auc_macro)

#             for i in classes:
#                 try:
#                     fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
#                     tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#                     split_tpr_sum[split][i] += tpr_interp
#                     split_count[split][i] += 1
#                     total_interp_points += len(split_fpr_fixed)
#                 except ValueError:
#                     continue

# # === Plot averaged ROC curves (1 plot for all splits) ===
# plt.figure(figsize=(8, 6))

# for split in ['train', 'val', 'test']:
#     for i in classes:
#         if split_count[split][i] > 0:
#             avg_tpr = split_tpr_sum[split][i] / split_count[split][i]
#             label = f'{split.capitalize()} - Class {i}' if n_classes > 2 else f'{split.capitalize()}'
#             plt.plot(split_fpr_fixed, avg_tpr, label=label, linewidth=2)

# # Style and format
# plt.xlabel("False Positive Rate", fontsize=20)
# plt.ylabel("True Positive Rate", fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(True)

# # Shift x-axis start to slightly right of 0
# plt.xlim([-0.03, 1.05])  # 👈 Adds space before 0.0
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(f"b_{task_type}_5.png")
# plt.close()

# print(f"\n✅ Total interpolated points across all seeds and splits: {total_interp_points}")

# # === Save Summary and JSON Results ===
# average_auc_scores = {
#     split: float(np.mean(scores)) if scores else None
#     for split, scores in split_auc_lists.items()
# }

# dict_df_out_forward = filter_dict_by_region(dict_df_out, region=None)
# dict_summary = compute_seed_metrics(
#     dict_df_out=dict_df_out_forward,
#     label_col=param_dict["label_col"],
#     pred_col=f"{param_dict['label_col']}_pred",
#     average='macro'
# )

# output_file = f"b_{task_type}_cb.json"
# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": best_params_dict,
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics": dict_summary,
#     "auc_scores": auc_results,
#     "average_auc_scores": average_auc_scores,
# }
# dict_to_json(final_dict, output_file)

# # === Optional: Show class distribution ===
# print_class_distributions(processed_df, label_col=param_dict["label_col"], split="val", expected_classes=classes)




#binary_xgboost_pipeline.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from copy import deepcopy

from zeolite_emt.data_transformation import (
    feat_dict,
    Transformation,
    PCA_Transformation,
    clean_discrete_columns,
    dict_to_json,
    json_to_dict,
    filter_dict_by_region,
)

from zeolite_emt.train_classifier import (
    dataset_preprocessing,
    dataset_split_df,
    XGB_grid_search,
    TrainXGB,
    compute_seed_metrics,
)

# Load config
with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

param_grid = config["param_grid_xgb"]
# === Choose task ===
#param_dict = config["ternary_classification_parameters"]
param_dict = config["binary_classification_parameters"]

# Dataset setup
df_main = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv'))
df_main["label_ter"] = df_main["label_ter"].replace({"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0})
df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})
continuous_feat_cols = feat_dict["continuous_feat_cols"]
discrete_feat_cols = feat_dict["discrete_feat_cols"]
input_feat_cols = continuous_feat_cols + discrete_feat_cols

# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [494,101,283,471,326]]

# Seed setup
# random_seed = 35
# np.random.seed(random_seed)
# seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()

#ternary
# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [77,273,486,490,101,494,471,283,336,28,326,155,86,17,281,136]]
#binary_reduced
# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [480, 155, 454, 176,292,163,205,315,136,474,101,212,80,77,117,471]]
#binary_all
np.random.seed(35)
seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
             if s not in [454,224,130,409,17,22,62,163,371,315,474,101,490,6,273,28]]
# Dataset split
df_process = dataset_split_df(
    df=df_main,
    label_col=param_dict["label_col"],
    seed_list=seed_list,
    partition=True,
    partition_col="Region",
    test_size=0.204,
    val_size=0.475,
)

df_forward = filter_dict_by_region(df_process, region=None)
#print(df_forward[77].shape)

# Preprocess
processed_df, args_dict = dataset_preprocessing(
    df_forward,
    continuous_feat_cols,
    input_feat_cols,
    transformation_type=param_dict["transformation"],
    pca=False,
    pca_dim=None,
)

# Grid search
xgb_grid_search = XGB_grid_search(
    param_grid=param_grid,
    seed_list=seed_list,
    feat_cols=input_feat_cols,
    label_col=param_dict["label_col"],
    objective=param_dict["objective"],
    num_class=param_dict["num_class"],
    eval_metric=param_dict["eval_metric"],
    average=param_dict["average"],
)
best_params_dict = xgb_grid_search.best_grid_evaluation(dict_df=processed_df)
print("Best Parameters:", best_params_dict)

# Train
train_xgb = TrainXGB(
    params=best_params_dict,
    seed_list=seed_list,
    feat_cols=input_feat_cols,
    label_col=param_dict["label_col"],
    objective=param_dict["objective"],
    num_class=param_dict["num_class"],
    eval_metric=param_dict["eval_metric"],
)
dict_df_out = train_xgb.set_up_XGB(processed_df)

# ROC AUC & Curve
n_classes = param_dict["num_class"]
classes = list(range(n_classes))
split_fpr_fixed = np.linspace(0, 1, 101)
split_tpr_sum = {split: {cls: np.zeros_like(split_fpr_fixed) for cls in classes} for split in ['train', 'val', 'test']}
split_count = {split: {cls: 0 for cls in classes} for split in ['train', 'val', 'test']}
auc_results = {}
split_auc_lists = {"train": [], "val": [], "test": []}
total_interp_points = 0

for seed in seed_list:
    df = dict_df_out[seed]
    auc_results[seed] = {}

    for split in ['train', 'val', 'test']:
        df_split = df[df["Split"] == split]
        if df_split.empty:
            continue

        y_true = df_split[param_dict["label_col"]]
        proba_col = f"{param_dict['label_col']}_proba"
        if proba_col not in df_split.columns:
            continue

        y_proba = np.array(df_split[proba_col].tolist())

        if n_classes == 2:
            y_proba_bin = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            auc = roc_auc_score(y_true, y_proba_bin)
            auc_results[seed][split] = {"auc": float(auc)}
            split_auc_lists[split].append(auc)

            fpr, tpr, _ = roc_curve(y_true, y_proba_bin)
            tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
            split_tpr_sum[split][0] += tpr_interp
            split_count[split][0] += 1
            total_interp_points += len(split_fpr_fixed)
        else:
            y_true_bin = label_binarize(y_true, classes=classes)
            auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            auc_results[seed][split] = {"macro_auc": float(auc_macro)}
            split_auc_lists[split].append(auc_macro)

            for i in classes:
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
                    split_tpr_sum[split][i] += tpr_interp
                    split_count[split][i] += 1
                    total_interp_points += len(split_fpr_fixed)
                except ValueError:
                    continue

# Plot ROC curves
plt.figure(figsize=(8, 6))
for split in ['train', 'val', 'test']:
    for i in classes:
        if split_count[split][i] > 0:
            avg_tpr = split_tpr_sum[split][i] / split_count[split][i]
            label = f'{split.capitalize()} - Class {i}' if n_classes > 2 else f'{split.capitalize()}'
            plt.plot(split_fpr_fixed, avg_tpr, label=label, linewidth=2)

plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.xlim([-0.03, 1.05])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig("b_xgb.png")
plt.close()
print(f"\n✅ Total interpolated points across all seeds and splits: {total_interp_points}")

# Save metrics
dict_df_out_forward = filter_dict_by_region(dict_df_out, region=None)
#print(dict_df_out_forward[77].shape)
dict_summary = compute_seed_metrics(
    dict_df_out=dict_df_out_forward,
    label_col=param_dict["label_col"],
    pred_col=f"{param_dict['label_col']}_pred",
    average='macro'
)

average_auc_scores = {
    split: float(np.mean(scores)) if scores else None
    for split, scores in split_auc_lists.items()
}

final_dict = {
    "summary_df": dict_df_out,
    "seed_list": seed_list,
    "best_params": best_params_dict,
    "processed_df": df_process,
    "transformation_args": args_dict,
    "statistics": dict_summary,
    "auc_scores": auc_results,
    "average_auc_scores": average_auc_scores,
}
dict_to_json(final_dict, "b_xgb.json")



# # tabpfn_pipeline.py
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import yaml
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.preprocessing import label_binarize

# from zeolite_emt.data_transformation import (
#     feat_dict,
#     filter_dict_by_region,
#     dict_to_json,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     TrainTabPFN,
#     compute_seed_metrics,
#     print_class_distributions,
# )

# # ==== CONFIG ====
# with open("params.yaml", "r") as f:
#     config = yaml.safe_load(f)

# task_type = "ternary"  # or "ternary"
# param_dict = config["binary_classification_parameters"] if task_type == "binary" else config["ternary_classification_parameters"]

# # ==== DATA ====
# df_main = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv'))
# df_main["label_ter"] = df_main["label_ter"].replace({"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0})
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})

# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols   = feat_dict["discrete_feat_cols"]
# input_feat_cols      = continuous_feat_cols + discrete_feat_cols

# np.random.seed(35)
# seed_list = [s for s in np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist() 
#              if s not in [77,273,486,490,101,494,471,283,336,28,326,155,86,17,281,136]]

# # ==== SEEDS & SPLIT ====
# # np.random.seed(35)
# # seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()

# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=False,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )

# df_forward = filter_dict_by_region(df_process, region=None)  # or None for entire
# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )

# # ==== TRAIN TABPFN ====
# train_tabpfn = TrainTabPFN(
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
# )
# dict_df_out = train_tabpfn.set_up_TabPFN(processed_df)

# # ==== ROC / AUC (same averaging logic) ====
# n_classes = param_dict["num_class"]
# classes   = list(range(n_classes))
# split_fpr_fixed = np.linspace(0, 1, 101)

# split_tpr_sum  = {s: {c: np.zeros_like(split_fpr_fixed) for c in classes} for s in ['train','val','test']}
# split_count    = {s: {c: 0 for c in classes} for s in ['train','val','test']}
# auc_results    = {}
# split_auc_lists = {"train": [], "val": [], "test": []}
# total_interp_points = 0

# for seed in seed_list:
#     df = dict_df_out[seed]
#     auc_results[seed] = {}

#     for split in ['train','val','test']:
#         df_split = df[df["Split"] == split]
#         if df_split.empty:
#             continue

#         y_true = df_split[param_dict["label_col"]]
#         proba_col = f"{param_dict['label_col']}_proba"
#         if proba_col not in df_split.columns:
#             continue

#         y_proba = np.array(df_split[proba_col].tolist())

#         if n_classes == 2:
#             # binary
#             y_proba_bin = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
#             auc = roc_auc_score(y_true, y_proba_bin)
#             auc_results[seed][split] = {"auc": float(auc)}
#             split_auc_lists[split].append(auc)

#             fpr, tpr, _ = roc_curve(y_true, y_proba_bin)
#             tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#             split_tpr_sum[split][0] += tpr_interp
#             split_count[split][0]   += 1
#             total_interp_points     += len(split_fpr_fixed)
#         else:
#             # multiclass
#             y_true_bin = label_binarize(y_true, classes=classes)
#             auc_macro  = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
#             auc_results[seed][split] = {"macro_auc": float(auc_macro)}
#             split_auc_lists[split].append(auc_macro)

#             for i in classes:
#                 try:
#                     fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
#                     tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#                     split_tpr_sum[split][i] += tpr_interp
#                     split_count[split][i]   += 1
#                     total_interp_points     += len(split_fpr_fixed)
#                 except ValueError:
#                     # class i absent
#                     continue

# # ==== PLOT (one figure, all splits) ====
# plt.figure(figsize=(8, 6))
# for split in ['train','val','test']:
#     for i in classes:
#         if split_count[split][i] > 0:
#             avg_tpr = split_tpr_sum[split][i] / split_count[split][i]
#             label = f'{split.capitalize()} - Class {i}' if n_classes > 2 else f'{split.capitalize()}'
#             plt.plot(split_fpr_fixed, avg_tpr, label=label, linewidth=2)

# plt.xlabel("False Positive Rate", fontsize=20)
# plt.ylabel("True Positive Rate",  fontsize=20)
# plt.xticks(fontsize=20); plt.yticks(fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(True)
# plt.xlim([-0.03, 1.05]); plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(f"tabpfn_{task_type}_test.png")
# plt.close()

# print(f"\n✅ Total interpolated points across all seeds/splits: {total_interp_points}")

# # ==== METRICS SUMMARY & SAVE ====
# average_auc_scores = {
#     split: float(np.mean(scores)) if scores else None
#     for split, scores in split_auc_lists.items()
# }

# dict_df_out_forward = filter_dict_by_region(dict_df_out, region=None)
# dict_summary = compute_seed_metrics(
#     dict_df_out=dict_df_out_forward,
#     label_col=param_dict["label_col"],
#     pred_col=f"{param_dict['label_col']}_pred",
#     average='macro'
# )

# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": None,  # TabPFN has no grid search
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics": dict_summary,
#     "auc_scores": auc_results,
#     "average_auc_scores": average_auc_scores,
# }
# dict_to_json(final_dict, f"{task_type}_tabpfn_test.json")

# # (optional) check class balance
# classes_expected = classes
# print_class_distributions(processed_df, label_col=param_dict["label_col"], split="val", expected_classes=classes_expected)















#lightgbm_pipeline.py
#lightgbm_pipeline.py
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import yaml
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.preprocessing import label_binarize
# from lightgbm import LGBMClassifier

# from zeolite_emt.data_transformation import (
#     feat_dict,
#     Transformation,
#     PCA_Transformation,
#     clean_discrete_columns,
#     dict_to_json,
#     json_to_dict,
#     filter_dict_by_region,
# )
# from zeolite_emt.train_classifier import (
#     dataset_preprocessing,
#     dataset_split_df,
#     LightGBMGridSearch,
#     TrainLightGBM,
#     compute_seed_metrics,
#     print_class_distributions,
# )

# # === Load Config ===
# with open("params.yaml", "r") as file:
#     config = yaml.safe_load(file)

# # === Task Type ===
# task_type = "ternary"  # or "binary"
# param_dict = config["binary_classification_parameters"] if task_type == "binary" else config["ternary_classification_parameters"]
# param_grid_lgb = config["param_grid_lightgbm"]

# # === Load Dataset ===
# df_main = pd.read_csv(os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv'))
# df_main["label_ter"] = df_main["label_ter"].replace({"pure-EMT": 2, "hybrid-EMT": 1, "non-EMT": 0})
# df_main["label_bin"] = df_main["label_bin"].replace({"pure-EMT": 1, "others": 0})

# continuous_feat_cols = feat_dict["continuous_feat_cols"]
# discrete_feat_cols = feat_dict["discrete_feat_cols"]
# input_feat_cols = continuous_feat_cols + discrete_feat_cols

# # === Seeds ===
# np.random.seed(35)
# seed_list = np.random.choice(500, size=param_dict["seed_size"], replace=False).tolist()

# # === Dataset Split & Preprocessing ===
# df_process = dataset_split_df(
#     df=df_main,
#     label_col=param_dict["label_col"],
#     seed_list=seed_list,
#     partition=False,
#     partition_col="Region",
#     test_size=0.204,
#     val_size=0.475,
# )

# df_forward = filter_dict_by_region(df_process, region=None)
# processed_df, args_dict = dataset_preprocessing(
#     df_forward,
#     continuous_feat_cols,
#     input_feat_cols,
#     transformation_type=param_dict["transformation"],
#     pca=False,
#     pca_dim=None,
# )

# # === Sanitize Columns ===
# def sanitize_columns(columns):
#     return [col.replace(':', '_')
#                 .replace('/', '_')
#                 .replace('(', '')
#                 .replace(')', '')
#                 .replace('²', '2')
#                 .replace('₃', '3')
#                 .replace(' ', '_')
#             for col in columns]

# input_feat_cols = sanitize_columns(input_feat_cols)
# processed_df = {
#     seed: df.rename(columns={old: new for old, new in zip(df.columns, sanitize_columns(df.columns))})
#     for seed, df in processed_df.items()
# }

# # === Grid Search ===
# lgb_grid_search = LightGBMGridSearch(
#     param_grid=param_grid_lgb,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
#     average=param_dict["average"],
# )
# best_params_dict = lgb_grid_search.best_grid_evaluation(dict_df=processed_df)
# print("Best Parameters:", best_params_dict)

# # === Train LightGBM ===
# train_lgb = TrainLightGBM(
#     params=best_params_dict,
#     seed_list=seed_list,
#     feat_cols=input_feat_cols,
#     label_col=param_dict["label_col"],
# )
# dict_df_out = train_lgb.set_up_lightgbm(processed_df)

# # === ROC Averaging and Plotting ===
# n_classes = param_dict["num_class"]
# classes = list(range(n_classes))
# split_fpr_fixed = np.linspace(0, 1, 101)

# split_tpr_sum = {split: {cls: np.zeros_like(split_fpr_fixed) for cls in classes} for split in ['train', 'val', 'test']}
# split_count = {split: {cls: 0 for cls in classes} for split in ['train', 'val', 'test']}
# auc_results = {}
# split_auc_lists = {"train": [], "val": [], "test": []}
# total_interp_points = 0

# for seed in seed_list:
#     df = dict_df_out[seed]
#     auc_results[seed] = {}

#     for split in ['train', 'val', 'test']:
#         df_split = df[df["Split"] == split]
#         if df_split.empty:
#             continue

#         y_true = df_split[param_dict["label_col"]]
#         y_proba = np.array(df_split[f"{param_dict['label_col']}_proba"].tolist())

#         if n_classes == 2:
#             auc = roc_auc_score(y_true, y_proba)
#             auc_results[seed][split] = {"auc": float(auc)}
#             split_auc_lists[split].append(auc)

#             fpr, tpr, _ = roc_curve(y_true, y_proba)
#             tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#             split_tpr_sum[split][0] += tpr_interp
#             split_count[split][0] += 1
#             total_interp_points += len(split_fpr_fixed)

#         else:
#             y_true_bin = label_binarize(y_true, classes=classes)
#             auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
#             auc_results[seed][split] = {"macro_auc": float(auc_macro)}
#             split_auc_lists[split].append(auc_macro)

#             for i in classes:
#                 try:
#                     fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
#                     tpr_interp = np.interp(split_fpr_fixed, fpr, tpr)
#                     split_tpr_sum[split][i] += tpr_interp
#                     split_count[split][i] += 1
#                     total_interp_points += len(split_fpr_fixed)
#                 except ValueError:
#                     continue

# # === Plot averaged ROC curves (1 figure, all splits) ===
# plt.figure(figsize=(7, 6))
# for split in ['train', 'val', 'test']:
#     for i in classes:
#         if split_count[split][i] > 0:
#             avg_tpr = split_tpr_sum[split][i] / split_count[split][i]
#             label = f'{split.capitalize()} - Class {i}' if n_classes > 2 else f'{split.capitalize()}'
#             plt.plot(split_fpr_fixed, avg_tpr, label=label)

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Averaged ROC Curves Across Seeds")
# plt.legend()
# plt.grid(True)
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(f"roc_average_all_splits_{task_type}_lightgbm.png")
# plt.close()

# print(f"\n✅ Total interpolated points across all seeds and splits: {total_interp_points}")

# # === Save Results ===
# average_auc_scores = {
#     split: float(np.mean(scores)) if scores else None
#     for split, scores in split_auc_lists.items()
# }

# dict_df_out_forward = filter_dict_by_region(dict_df_out, region='in')
# dict_summary = compute_seed_metrics(
#     dict_df_out=dict_df_out_forward,
#     label_col=param_dict["label_col"],
#     pred_col=f"{param_dict['label_col']}_pred",
#     average='macro'
# )

# output_file = f"{task_type}_lightgbm_results.json"
# final_dict = {
#     "summary_df": dict_df_out,
#     "seed_list": seed_list,
#     "best_params": best_params_dict,
#     "processed_df": df_process,
#     "transformation_args": args_dict,
#     "statistics": dict_summary,
#     "auc_scores": auc_results,
#     "average_auc_scores": average_auc_scores,
# }
# dict_to_json(final_dict, output_file)

# print_class_distributions(processed_df, label_col=param_dict["label_col"], split="val", expected_classes=classes)
