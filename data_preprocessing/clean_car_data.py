import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def replace_word(data):
    for col in ('buying', 'maint'):
        data[col] = data[col].replace('vhigh', 'high')

    return data

'''def write_file(data, fileName):
    new_data = pd.DataFrame(data=data).to_csv(fileName)'''

if __name__ == '__main__':
    filePath = "./../car.csv"

    data = open_file(filePath)
    targets = np.array(data)[:, -1]
    replace_word(data)
    #print(data['buying'])
    #print(data['maint'])