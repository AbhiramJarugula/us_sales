import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


def eda_before(dataset):
    print("=====Exploratory Data Analysis Before Data Preprocessing=====")
    print("=====First 5 Rows=====")
    print(dataset.head(5))
    print("=====Last 5 Rows=====")
    print(dataset.tail(5))
    print("=====Random Sample 5 Rows=====")
    print(dataset.sample(5))

    print("=====Dataset Dimensions=====")
    cols = dataset.columns
    print("No.of Columns:", len(cols))
    n_rows = dataset.shape[0]
    print("No.of Rows:", n_rows)
    print("rows*columns:", dataset.shape)

    print("=====Column Names=====")
    for index, col in enumerate(cols):
        print("index:", index, "Column:", col)
        print("\n")

    print("=====Basic Info About Datset=====")
    print(dataset.info())

    print("=====Statistical Info about the dataset for all the columns=====")
    print(dataset.describe(include='all'))

    print("=====Null Values=====")
    print(f"Null values per col:\n{dataset.isnull().sum()}\n")
    print(f"Percentage of null values in the dataset:\n{dataset.isnull().mean() * 100}\n")

    print("=====duplicate values=====")
    print(f"no.of duplicate values:\n{dataset.duplicated().sum()}\n")

    print("=====unique values=====")
    print(f"no.of unique values per column:\n{dataset.nunique()}\n")
    for col in cols:
        print("\n")
        print(f"{col}:{dataset[col].unique()}\n")

    print("=====low confidence limit min value=====")
    print("the minimum value in the low confidence limit")
    print(min(dataset['Low_Confidence_Limit']))

    print("=====high confidence limit min value=====")
    print("the minimum value in the high confidence limit")
    print(min(dataset['High_Confidence_Limit']))

    print("=====low confidence limit max value=====")
    print("the maximum value in the low confidence limit")
    print(max(dataset['Low_Confidence_Limit']))

    print("=====high confidence limit max value=====")
    print("the maximum value in the high confidence limit")
    print(max(dataset['High_Confidence_Limit']))

    print("=====Mean Value of the low confidence limit=====")
    print(np.average(dataset['Low_Confidence_Limit']))

    print("=====Mean Value of the High confidence limit=====")
    print(np.average(dataset['High_Confidence_Limit']))