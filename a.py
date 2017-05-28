import pandas as pd
import numpy as np
import xgboost as xgboost

from pandas.tools.plotting import scatter_matrix

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data


def outliers_treatment():
    return data




if __name__ == '__main__':
    filePath = "train.csv"
    data = open_file(filePath)
    targets = np.array(data)[:, -1]
