from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import logging

logger = logging.getLogger(__name__)
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
        temp = numpy_data[:, i]
        print(numpy_data[:, i])
        dict = np.unique(numpy_data[:, i])
        print("---------------------------------------------")
        print(i)
        print(dict)
        print("---------------------------------------------")
        if type(dict[0]) == str:
            for j in range(len(dict)):
                temp[np.where(numpy_data[:, i] == dict[j])] = j
            numpy_data[:, i] = temp
    return numpy_data

#LIMPIEZA DE OUTLIERS

def remove_outliers(data, feature, outlier_value):
    outliers = data.iloc[data[feature] >= outlier_value, feature].index
    data.drop(outliers, inplace=True)
    return data


#APLICANDO NORMALIZACION

def min_max_scaler(data):
    """# import data
    num_features = len(data.columns) - 1
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))
    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])
    features = data[list(range(1, num_features))]
    target = data[[num_features]]"""

    features = data[:,0:-1]
    target = data[:,-1]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(features)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data



#ATTRIBUTTE SUBSET SELECTION
def attribute_subset_selection_with_trees(data, type):
    features = data.iloc[:, 0:-1]
    target = data.iloc[:, -1]
    feature_vector = features
    targets = target
    """
    This function use gain info to select the best features of a numpy array
    :param type: Classification or Regression
    :param feature_vector: Numpy array to be transformed
    :param targets: feature vector targets
    :return: Numpy array with the selected features
    """
    if type == 'Regression':
        extra_tree = ExtraTreesRegressor()
    else:
        # Model declaration
        extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(feature_vector, targets)

    # Model information:
    logger.debug('Model information:')

    # display the relative importance of each attribute
    logger.debug('Importance of every feature: %s', extra_tree.feature_importances_)

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit=True)

    # Model transformation
    new_feature_vector = model.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])
    return new_feature_vector


if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/testTrain4.csv")
    #show_data_info(data)


    #PRIMERA ITERACION
    replace_missing_values_with_constant(data['build_year'], "-1")
    replace_missing_values_with_mode(data, ['life_sq', 'floor', 'max_floor', 'kitch_sq', 'state', 'num_room', 'material', 'railroad_station_walk_km', 'metro_min_walk',
                                            'hospital_beds_raion', 'metro_km_walk'])
    # replace_mv_with_constant(data, -1)

    # convert_data_to_numeric(data)

    #SEGUNDA ITERACION
    # remove_outliers(data, 'full_sq', 5000)
    # remove_outliers(data, 'life_sq', 7000)
    # remove_outliers(data, 'floor', 70)
    # remove_outliers(data, 'max_floor', 100)
    # remove_outliers(data, 'kitch_sq', 1700)
    # remove_outliers(data, 'school_km', 50)
    # remove_outliers(data, 'life_sq', 40)
    # remove_outliers(data, 'industrial_km', 20)
    # remove_outliers(data, 'school_km', 40)
    # remove_outliers(data, 'mosque_km', 40)
    # min_max_scaler(data)
    # attribute_subset_selection_with_trees(data, "Regression")

    print(data)