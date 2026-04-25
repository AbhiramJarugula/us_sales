import os as os

from config import DATASET_PATH
from dataloading import dataloading
from exploratory_data_analysis import eda_before
from datapreprocessing import preprocess_data

if __name__ == '__main__':
    print("Application Start Successfull")

    dataset = dataloading(DATASET_PATH)

    eda_before(dataset)

    dataframe = preprocess_data(dataset)