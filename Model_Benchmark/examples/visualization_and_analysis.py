import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,sys
from zeolite_emt.data_transformation import (Transformation,feat_dict)



def distribution_plot(
    df_transformed,
    df_orig,
    col,
    transform_type="Std",
    bins=100,
    fontsize=13,
    figsize=(4, 6),
    kde=True,
    stat="density",
):

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Plot for transformed data
    sns.histplot(data=df_transformed, x=col, kde=kde, stat=stat, bins=bins, ax=axs[0])
    axs[0].set_title(
        str(transform_type) + " transformed " + str(col) + " distribution", fontsize=fontsize
    )
    axs[0].set_ylabel("Density", fontsize=fontsize)
    axs[0].tick_params(axis="both", labelsize=fontsize)

    # Plot for original data
    sns.histplot(data=df_orig, x=col, kde=kde, stat=stat, bins=bins, ax=axs[1])
    axs[1].set_title("Original " + str(col) + " distribution", fontsize=fontsize)
    axs[1].set_xlabel(str(col), fontsize=fontsize)
    axs[1].set_ylabel("Density", fontsize=fontsize)
    axs[1].tick_params(axis="both", labelsize=fontsize)

    plt.tight_layout()

    plt.savefig(str(transform_type)
        + "_transformed_histogram_"
        + str(col).replace("/", "_")
        + ".png",
        dpi=800,
        bbox_inches="tight",
    )
    plt.close()

def setting_up_transformation(df_in,feat_dict,transform_type="std"):
    transform_cols=feat_dict["continuous_feat_cols"]
    transform = Transformation(df_in, transform_cols)
    if transform_type in ["std","min_max"]:
        if transform_type == "std":
            df_transformed = transform.std_transform()
        else:
            df_transformed = transform.min_max_transform()
    else:
        print("Only std or min_max allowed, defaulting to std transform")
        df_transformed = transform.std_transform()
    for col in transform_cols:
        print(col)
        print("Mean: ", df_transformed[col].mean(), "std: ", df_transformed[col].std())
        distribution_plot(df_transformed, df_in, col=col, transform_type=transform_type)


processed_file = os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv')

orig_df = pd.read_csv(processed_file)
for transform_type in ["std","min_max"]:
    setting_up_transformation(orig_df,feat_dict,transform_type=transform_type)