import os,sys
import pandas as pd
import numpy as np
from copy import deepcopy

# Path of original dataset
orig_dataset_file = os.path.join(os.path.dirname(__file__), '../dataset/zeolites_dataset.csv')
processed_file = os.path.join(os.path.dirname(__file__), '../dataset/processed_dataset.csv')

orig_df = pd.read_csv(orig_dataset_file)
##### Processing the dataset to get numeric features

# Step 1: Our dataset has five (two) different reagants for silicon (aluminium) and here we are expanding
#           Si source column to five and Al source column to 2.

df_expanded = pd.get_dummies(
    orig_df, columns=["Si source", "Al source"], drop_first=False
)

# Step 2: Transforming boolen type (yes/no/true/false) to 0 and 1. O for no/false and 1 for yes/true
df_numeric = df_expanded.replace(
    {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
)


# Step 3: Defining and writting ternary labels, "pure-EMT", "hybrid-EMT" and "non-EMT" in the dataset
def classify_emt_ternary(product):
    product = str(
        product
    ).strip()  # ensure it's a string and remove leading/trailing spaces
    if product == "EMT":
        return "pure-EMT"
    elif "EMT" in product:
        return "hybrid-EMT"
    else:
        return "non-EMT"


df_numeric["label_ter"] = df_numeric["Product"].apply(classify_emt_ternary)


# Step 4: Defining and writting binary labels, "pure-EMT", and "others" in the dataset
def classify_emt_binary(product):
    product = str(
        product
    ).strip()  # ensure it's a string and remove leading/trailing spaces
    if product == "EMT":
        return "pure-EMT"
    else:
        return "others"


df_numeric["label_bin"] = df_numeric["Product"].apply(classify_emt_binary)

# Step 5: Removing unwanted columns

df_numeric = df_numeric.drop(columns=["Entry", "Product"])


# Step 6: Renaming some columns for better readability
df_numeric.rename(
    columns={
        "Si precursor sizs (nm)": "Si:prec",
        "Pre-dissolution of Si": "Pd:Si",
        "Pre-dissolution of Al": "Pd:Al",
        "Time (h)": "Time",
        "Temperature (°C)": "Temp",
        "Si source_Fumed silica": "Si:fumed-silica",
        "Si source_Ludox-AS40": "Si:Ludox-AS40",  # remains unchanged
        "Si source_Ludox-SM30": "Si:Ludox-SM30",
        "Si source_Na2SiO3": "Si:Na2SiO3",
        "Si source_TEOS": "Si:TEOS",
        "Al source_Al(OH)3": "Al:Al(OH)3",
        "Al source_NaAlO2": "Al:NaAlO2",
    },
    inplace=True,
)


new_order = [
    "ID",
    "Na",
    "Al",
    "Si",
    "H2O",
    "Si/OH",
    "Si/Al",
    "Time",
    "Temp",
    "Si:prec",
    "Pd:Si",
    "Pd:Al",
    "Al:Al(OH)3",
    "Al:NaAlO2",
    "Si:fumed-silica",
    "Si:Ludox-AS40",
    "Si:Ludox-SM30",
    "Si:Na2SiO3",
    "Si:TEOS",
    "label_ter",
    "label_bin",
]
df_final = df_numeric[new_order]

# Step 7: Displaying distribution of ternary labels
print(df_final["label_ter"].value_counts())

# Step 8: For some analysis in the future, for Na values between 50 and 30 and for Si/Al values between 2.5 to 9, datapoints are labelled as 'in'
#       while the remaining as 'out'

mask = (
        (df_final["Na"] < 50) & (df_final["Na"] >= 30) &
        (df_final["Si/Al"] <= 9) & (df_final["Si/Al"] >= 2.5)
)
df_fin=deepcopy(df_final)
df_fin["Region"] = np.where(mask, "in", "out")

# Step 9: Writting the final processed dataset
if os.path.exists(processed_file):
    print("File already exists, proceeding to delete")
    os.remove(processed_file)
df_fin.to_csv(processed_file, index=False)
