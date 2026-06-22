from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import pandas as pd
import numpy as np


class Transformation_single:
    def __init__(self, df_in, transform_cols):
        """
        Initialize with the dataframe and the columns to be transformed.

        Args:
            df_in (pd.DataFrame): Input DataFrame with columns to be transformed.
            transform_cols (list): List of column names to apply standard scaling.
        """
        self.df_in = df_in
        self.cols = transform_cols
        self.std_scaler = (
            StandardScaler()
        )  # Store std_scaler for inverse transform if needed
        self.minmax_scaler = (
            MinMaxScaler()
        )  # Store MinMax_scaler for inverse transform if needed

    def std_transform(self):
        """
        Applies standard scaling to selected columns while preserving the original column order.

        Returns:
            pd.DataFrame: Transformed DataFrame with scaled and unscaled columns in original order.
        """
        df = deepcopy(self.df_in)
        df_to_scale = df[self.cols]
        df_rest = df.drop(columns=self.cols)

        df_std = pd.DataFrame(
            self.std_scaler.fit_transform(df_to_scale),
            columns=self.cols,
            index=df.index,
        )

        # Combine and reorder to original column order
        df_transformed = pd.concat([df_std, df_rest], axis=1)
        df_transformed = df_transformed[self.df_in.columns]

        return df_transformed

    def inv_std_transform(self, df_transformed):
        """
        Inversely transforms the scaled columns using the fitted scaler.

        Args:
            df_transformed (pd.DataFrame): DataFrame with standardized values in `transform_cols`.

        Returns:
            pd.DataFrame: DataFrame with inverse-transformed `transform_cols` and original column order.
        """
        df_scaled_cols = df_transformed[self.cols]
        df_rest = df_transformed.drop(columns=self.cols)

        df_in_scale = pd.DataFrame(
            self.std_scaler.inverse_transform(df_scaled_cols),
            columns=self.cols,
            index=df_transformed.index,
        )

        df_in_space = pd.concat([df_in_scale, df_rest], axis=1)
        df_in_space = df_in_space[self.df_in.columns]

        return df_in_space

    def min_max_transform(self):
        """
        Applies min-max scaling (0 to 1) to selected columns while preserving the original column order.

        Returns:
            pd.DataFrame: Transformed DataFrame with scaled and unscaled columns in original order.
        """
        df = deepcopy(self.df_in)
        df_to_scale = df[self.cols]
        df_rest = df.drop(columns=self.cols)

        df_minmax = pd.DataFrame(
            self.minmax_scaler.fit_transform(df_to_scale),
            columns=self.cols,
            index=df.index,
        )

        df_transformed = pd.concat([df_minmax, df_rest], axis=1)
        df_transformed = df_transformed[self.df_in.columns]

        return df_transformed

    def inv_min_max_transform(self, df_transformed):
        """
        Inversely transforms the min-max scaled columns using the fitted scaler.

        Args:
            df_transformed (pd.DataFrame): DataFrame with min-max scaled values in `transform_cols`.

        Returns:
            pd.DataFrame: DataFrame with inverse-transformed `transform_cols` and original column order.
        """
        df_scaled_cols = df_transformed[self.cols]
        df_rest = df_transformed.drop(columns=self.cols)

        df_in_scale = pd.DataFrame(
            self.minmax_scaler.inverse_transform(df_scaled_cols),
            columns=self.cols,
            index=df_transformed.index,
        )

        df_in_space = pd.concat([df_in_scale, df_rest], axis=1)
        df_in_space = df_in_space[self.df_orig.columns]

        return df_in_space


class PCA_Transformation_single:
    def __init__(self, df_in, input_cols, pca_dim):
        self.df_in = df_in
        self.input_cols = input_cols
        self.pca_dim = pca_dim
        self.pca = None
        self.X_pca = None
        self.PCA_cols = [f"PCA{i+1}" for i in range(pca_dim)]
        self.df_pca = None
        self.explained_variance = None
        self.cumulative_explained_variance = None

    def pca_transform(self):
        df_transformed = deepcopy(self.df_in)
        X_orig = df_transformed[self.input_cols].to_numpy()

        self.pca = PCA(n_components=self.pca_dim)
        self.X_pca = self.pca.fit_transform(X_orig)
        self.explained_variance = self.pca.explained_variance_ratio_
        self.cumulative_explained_variance = np.cumsum(self.explained_variance)
        df_pca_components = pd.DataFrame(
            self.X_pca, columns=self.PCA_cols, index=self.df_in.index
        )
        print(df_pca_components.head())
        # Drop original input columns and insert PCA columns
        df_transformed.drop(columns=self.input_cols, inplace=True)
        df_transformed[self.PCA_cols] = df_pca_components

        self.df_pca = df_transformed
        return df_transformed

    def inv_pca_transform(self, df_with_pca=None):
        """
        Inverse transform PCA components back to original feature space.

        Args:
            df_with_pca (pd.DataFrame): Optional external dataframe with PCA components.
                                        If None, uses internally stored PCA-transformed data.
        Returns:
            pd.DataFrame: DataFrame with original input features restored.
        """
        if self.pca is None:
            raise RuntimeError(
                "PCA model has not been fitted. Run `pca_transform()` first."
            )

        if df_with_pca is None:
            if self.df_pca is None or self.X_pca is None:
                raise RuntimeError("No internal PCA-transformed data found.")
            df_pca_source = self.df_pca
            X_pca_input = self.X_pca
        else:
            df_pca_source = deepcopy(df_with_pca)
            try:
                X_pca_input = df_pca_source[self.PCA_cols].to_numpy()
            except KeyError:
                raise ValueError(
                    f"Provided DataFrame must contain PCA columns: {self.PCA_cols}"
                )

        # Inverse transform
        X_reconstructed = self.pca.inverse_transform(X_pca_input)

        # Drop PCA columns and restore original input columns
        df_in_space= df_pca_source.drop(columns=self.PCA_cols)
        df_in_space[self.input_cols] = pd.DataFrame(
            X_reconstructed, columns=self.input_cols, index=df_in_space.index
        )

        return df_in_space



class Transformation:
    def __init__(self, dict_df_in, transform_cols):
        """
        Initialize with a dictionary of DataFrames and the columns to be transformed.

        Args:
            dict_df_in (dict): Dictionary of {seed: DataFrame}.
            transform_cols (list): List of column names to apply scaling.
        """
        self.dict_df_in = dict_df_in
        self.cols = transform_cols
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def std_transform(self):
        """
        Applies standard scaling to selected columns in each dataframe while preserving column order.

        Returns:
            dict: Dictionary of transformed DataFrames.
        """
        transformed_dict = {}
        for seed, df in self.dict_df_in.items():
            df_copy = deepcopy(df)
            df_to_scale = df_copy[self.cols]
            df_rest = df_copy.drop(columns=self.cols)

            df_std = pd.DataFrame(
                self.std_scaler.fit_transform(df_to_scale),
                columns=self.cols,
                index=df.index,
            )

            df_transformed = pd.concat([df_std, df_rest], axis=1)
            df_transformed = df_transformed[df.columns]  # Reorder to original
            transformed_dict[seed] = df_transformed
        return transformed_dict

    def inv_std_transform(self, dict_df_transformed):
        """
        Inversely transforms the scaled columns for each DataFrame using the fitted scaler.

        Args:
            dict_df_transformed (dict): Dictionary of standardized DataFrames.

        Returns:
            dict: Dictionary of inverse-transformed DataFrames.
        """
        inv_transformed_dict = {}
        for seed, df in dict_df_transformed.items():
            df_scaled_cols = df[self.cols]
            df_rest = df.drop(columns=self.cols)

            df_in_scale = pd.DataFrame(
                self.std_scaler.inverse_transform(df_scaled_cols),
                columns=self.cols,
                index=df.index,
            )

            df_in_space = pd.concat([df_in_scale, df_rest], axis=1)
            df_in_space = df_in_space[df.columns]  # Reorder to original
            inv_transformed_dict[seed] = df_in_space
        return inv_transformed_dict

    def min_max_transform(self):
        """
        Applies min-max scaling to selected columns in each dataframe while preserving column order.

        Returns:
            dict: Dictionary of transformed DataFrames.
        """
        transformed_dict = {}
        for seed, df in self.dict_df_in.items():
            df_copy = deepcopy(df)
            df_to_scale = df_copy[self.cols]
            df_rest = df_copy.drop(columns=self.cols)

            df_minmax = pd.DataFrame(
                self.minmax_scaler.fit_transform(df_to_scale),
                columns=self.cols,
                index=df.index,
            )

            df_transformed = pd.concat([df_minmax, df_rest], axis=1)
            df_transformed = df_transformed[df.columns]
            transformed_dict[seed] = df_transformed
        return transformed_dict

    def inv_min_max_transform(self, dict_df_transformed):
        """
        Inversely transforms the min-max scaled columns for each DataFrame using the fitted scaler.

        Args:
            dict_df_transformed (dict): Dictionary of min-max scaled DataFrames.

        Returns:
            dict: Dictionary of inverse-transformed DataFrames.
        """
        inv_transformed_dict = {}
        for seed, df in dict_df_transformed.items():
            df_scaled_cols = df[self.cols]
            df_rest = df.drop(columns=self.cols)

            df_in_scale = pd.DataFrame(
                self.minmax_scaler.inverse_transform(df_scaled_cols),
                columns=self.cols,
                index=df.index,
            )

            df_in_space = pd.concat([df_in_scale, df_rest], axis=1)
            df_in_space = df_in_space[df.columns]
            inv_transformed_dict[seed] = df_in_space
        return inv_transformed_dict


from sklearn.decomposition import PCA
from copy import deepcopy
import pandas as pd
import numpy as np

class PCA_Transformation:
    def __init__(self, dict_df_in, input_cols, pca_dim):
        """
        Initialize with dictionary of DataFrames and PCA settings.

        Args:
            dict_df_in (dict): Dictionary of {seed: DataFrame}.
            input_cols (list): List of column names to apply PCA.
            pca_dim (int): Number of principal components to retain.
        """
        self.dict_df_in = dict_df_in
        self.input_cols = input_cols
        self.pca_dim = pca_dim
        self.PCA_cols = [f"PCA{i+1}" for i in range(pca_dim)]

        self.pca = None
        self.X_pca = {}  # dict of seed -> transformed PCA array
        self.df_pca = {}  # dict of seed -> PCA-transformed dataframe
        self.explained_variance = None
        self.cumulative_explained_variance = None

    def pca_transform(self):
        """
        Fit PCA on concatenated data and transform each seed-specific DataFrame.

        Returns:
            dict: Dictionary of PCA-transformed DataFrames.
        """
        # Concatenate all input data for global PCA fit
        all_data = pd.concat(
            [df[self.input_cols] for df in self.dict_df_in.values()], axis=0
        )
        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(all_data)

        self.explained_variance = self.pca.explained_variance_ratio_
        self.cumulative_explained_variance = np.cumsum(self.explained_variance)

        # Transform each seed's dataframe
        transformed_dict = {}
        for seed, df in self.dict_df_in.items():
            df_copy = deepcopy(df)
            X_orig = df_copy[self.input_cols].to_numpy()
            X_pca = self.pca.transform(X_orig)
            self.X_pca[seed] = X_pca

            df_pca_components = pd.DataFrame(X_pca, columns=self.PCA_cols, index=df.index)
            df_copy.drop(columns=self.input_cols, inplace=True)
            df_copy[self.PCA_cols] = df_pca_components
            self.df_pca[seed] = df_copy
            transformed_dict[seed] = df_copy

        return transformed_dict

    def inv_pca_transform(self, dict_df_with_pca=None):
        """
        Inverse transform PCA components for each seed back to original feature space.

        Args:
            dict_df_with_pca (dict, optional): Dictionary of DataFrames with PCA columns.
                                               If None, uses internal df_pca + X_pca.

        Returns:
            dict: Dictionary of DataFrames with original input features restored.
        """
        if self.pca is None:
            raise RuntimeError("PCA model has not been fitted. Run `pca_transform()` first.")

        inv_transformed_dict = {}

        for seed in self.dict_df_in.keys():
            if dict_df_with_pca is None:
                if seed not in self.df_pca or seed not in self.X_pca:
                    raise RuntimeError(f"No PCA-transformed data found for seed {seed}.")
                df_pca_source = self.df_pca[seed]
                X_pca_input = self.X_pca[seed]
            else:
                df_pca_source = deepcopy(dict_df_with_pca[seed])
                try:
                    X_pca_input = df_pca_source[self.PCA_cols].to_numpy()
                except KeyError:
                    raise ValueError(
                        f"DataFrame for seed {seed} must contain PCA columns: {self.PCA_cols}"
                    )

            X_reconstructed = self.pca.inverse_transform(X_pca_input)

            df_in_space = df_pca_source.drop(columns=self.PCA_cols)
            df_in_space[self.input_cols] = pd.DataFrame(
                X_reconstructed, columns=self.input_cols, index=df_in_space.index
            )

            inv_transformed_dict[seed] = df_in_space

        return inv_transformed_dict

def clean_discrete_columns(df, *col_groups):
    """
    Clean discrete/categorical columns after inverse PCA.

    For each group of columns:
      - If the group has a single column, it is rounded to nearest 0 or 1.
      - If the group has multiple columns, argmax is used to set the max column to 1 and others to 0.

    Args:
        df (pd.DataFrame): DataFrame with columns to clean.
        *col_groups: Variable-length list of column names or lists of column names.
                     Each group is either a string (single column) or a list of column names (one-hot group).

    Returns:
        pd.DataFrame: Modified DataFrame with corrected discrete values.
    """
    df_cleaned = df.copy()

    for group in col_groups:
        if isinstance(group, str):
            # Single column → round to 0 or 1
            if group in df_cleaned.columns:
                df_cleaned[group] = df_cleaned[group].apply(lambda x: 1 if x >= 0.5 else 0)
        elif isinstance(group, (list, tuple)) and all(col in df_cleaned.columns for col in group):
            values = df_cleaned[group].to_numpy()
            max_indices = np.argmax(values, axis=1)
            one_hot = np.zeros_like(values)
            for i, idx in enumerate(max_indices):
                one_hot[i, idx] = 1
            df_cleaned[group] = one_hot
        else:
            raise ValueError(f"Invalid column group: {group}")

    return df_cleaned



feat_dict = {
    "continuous_feat_cols": [
        "Na",
        "Al",
        "Si",
        "H2O",
        "Si/OH",
        "Si/Al",
        "Time",
        "Temp",
        "Si:prec",
    ],
    "discrete_feat_cols": [
        "Pd:Si",
        "Pd:Al",
        "Al:Al(OH)3",
        "Al:NaAlO2",
        "Si:fumed-silica",
        "Si:Ludox-AS40",
        "Si:Ludox-SM30",
        "Si:Na2SiO3",
        "Si:TEOS",
    ],
}

from json_tricks import dump,load
def dict_to_json(param_dict,json_file_name):
    with open(json_file_name, "w") as f:
        dump(param_dict, f, indent=4)

def json_to_dict(json_file_name):
    with open(json_file_name, "r") as f:
        original_dict = load(f)
    return dict(original_dict)

def filter_dict_by_region(dict_df, region=None):
    """
    Filters each DataFrame in a dictionary to only include rows
    where the 'Region' column matches the specified value.

    Args:
        dict_df (dict): Dictionary of {seed: DataFrame}.
        region (str): Value to match in the 'Region' column (default: 'in').

    Returns:
        dict: New dictionary with filtered DataFrames.
    """
    filtered_dict = {}
    for seed, df in dict_df.items():
        if 'Region' in df.columns:
            if region is None:
                return dict_df
            else:
                filtered_df = df[df['Region'] == region].copy()
                filtered_dict[seed] = filtered_df
        else:
            raise ValueError(f"'Region' column not found in DataFrame for seed {seed}")
    return filtered_dict