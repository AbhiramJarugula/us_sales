import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dataloading(DATASET_PATH):
    print("Data Loading Started🟢")

    dataset = pd.read_parquet(DATASET_PATH)
    #dataset.to_csv('limited_data.csv', index=False)
    #dataset.to_parquet('limited_data.parquet', index=False)


    print("=====Dataset Dimensions=====")
    cols = dataset.columns
    print("No.of Columns:", len(cols))
    n_rows = dataset.shape[0]
    print("No.of Rows:", n_rows)
    print("rows*columns:", dataset.shape)

    print("=====Column Names=====")
    for index, col in enumerate(cols):
        print(index, col)

    print("Data Loading Completed🟢")

    return dataset