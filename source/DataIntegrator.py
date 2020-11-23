import json
import re

import pandas as pd
from scipy.io import arff
import numpy as np


def check_for_ohe(columns):
    # determine that <ohe>_ is not at the beginning of input columns
    for column in columns:
        pattern = re.compile("<ohe>_*")
        assert not pattern.match(column), "column names of the dataset shouldn't start with <ohe>_"


def read_in_data_csv(path, delimiter):
    dataframe = pd.read_csv(path, delimiter=delimiter)

    check_for_ohe(dataframe.columns)

    for column in dataframe.columns:
        if dataframe[column].dtype != "int64" and dataframe[column].dtype != "float64":
            dataframe[column] = dataframe[column].replace({"?": np.nan})

    for column in dataframe.columns:
        column_type = column.split(":", 1)[1].lower().strip()
        if column_type == "categorical":
            dataframe[column] = dataframe[column].astype("category")
        elif column_type == "numerical":
            dataframe[column] = dataframe[column].astype("float64")
        else:
            raise Exception("Column type should be \"numerical\" or \"categorical\"! -> see config! Also consider if maybe the wrong delimiter was used for CSV-files")

        new_column_name = column.split(":", 1)[0].lower().strip()
        dataframe.rename(columns={column: new_column_name}, inplace=True)

    return dataframe


def read_in_data_arff(path):
    dataset, meta = arff.loadarff(open(path))

    dataframe = pd.DataFrame(dataset)

    check_for_ohe(dataframe.columns)

    return dataframe


# read in and transform data to pandas-dataframe
def read_in_data(path, csv_delimiter):
    if re.search("\.arff", path):
        dataframe = read_in_data_arff(path)
    elif re.search("\.csv", path):
        dataframe = read_in_data_csv(path, csv_delimiter)
    else:
        raise Exception("File should be .csv or .arff")

    return dataframe