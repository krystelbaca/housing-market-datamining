import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(10))

    print("Numerical Information:\n")
    numerical_info = data.iloc[:, :data.shape[1]]
    print(numerical_info.describe())

def replace_missing_values_with_constant(data, constant):
   data.fillna(constant, inplace=True)
   return data

def replace_mv_with_constant(data, constant):
   data.fillna(constant, inplace=True)
   return data

def replace_missing_values_with_mode(data, features):
    features = data[features]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])
    return data

def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        print(numpy_data[:,i])
        dict = np.unique(numpy_data[:,i])
        print("---------------------------------------------")
        print(i)
        print(dict)
        print("---------------------------------------------")
        if type(dict[0]) == str:
            for j in range(len(dict)):
                temp[np.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:,i] = temp
    return numpy_data

def remove_outliers(data, feature, outlier_value):
    outliers = data.loc[data[feature] >= outlier_value, feature].index
    data.drop(outliers, inplace=True)
    return data


if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/train.csv")



    #show_data_info(data)


    #data = delete_missing_values(data, 'instance')
    replace_missing_values_with_constant(data['build_year'], "NoInfo")
    replace_missing_values_with_mode(data, ['floor', 'num_room', 'material', 'state', 'preschool_quota', 'school_quota', 'kitch_sq', 'max_floor'])
    replace_mv_with_constant(data, -1)
    remove_outliers(data, 'full_sq', 5000)
    remove_outliers(data, 'life_sq', 7000)
    remove_outliers(data, 'floor', 70)
    remove_outliers(data, 'max_floor', 100)
    remove_outliers(data, 'kitch_sq', 1700)
    remove_outliers(data, 'school_km', 50)
    remove_outliers(data, 'life_sq', 40)
    remove_outliers(data, 'industrial_km', 20)
    remove_outliers(data, 'school_km', 40)
    remove_outliers(data, 'mosque_km', 40)

    print(data)