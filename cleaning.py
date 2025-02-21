import output as output
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

import logging

logger = logging.getLogger(__name__)
def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

def write_file(data, file_name):
    new_data = pd.DataFrame(data=data).to_csv(file_name)

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0]))
    print("Number of fetures: " + str(data.shape[1]))

    print('------------------------------------------')

    print("Initial instances:\n")
    print(data.head(10))

    print("Numerical Information:\n")
    numerical_info = data.iloc[:, :data.shape[1]]
    print(numerical_info.describe())

#VALORES FALTANTES

def replace_missing_values_with_constant(data, constant):
   data.fillna(constant, inplace=True)
   return data

'''Funcion que quita todos los NaN de todo el DataSet'''
def replace_mv_with_constant(data, constant):
   data.fillna(constant, inplace=True)
   return data

def replace_missing_values_with_mode(data, features):
    features = data[features]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])
    return data



#LIMPIEZA DE OUTLIERS

def remove_outliers(data, feature, outlier_value):
    outliers = data.iloc[data[feature] >= outlier_value, feature].index
    data.drop(outliers, inplace=True)
    return data

def nominal_to_numeric(train_df):
   for f in train_df.columns:
       if train_df[f].dtype=='object':
           lbl = preprocessing.LabelEncoder()
           lbl.fit(list(train_df[f].values.astype('str')))
           train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

def convert_data_to_numeric(data):
        numpy_data = data.values

        for i in range(len(numpy_data[0])):
            temp = numpy_data[:, i]
            dict = pd.unique(numpy_data[:, i])
            # print(dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[np.where(numpy_data[:, i] == dict[j])] = j

            numpy_data[:, i] = temp

        return numpy_data

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

    features = data.iloc[:, 0:-1]
    target = data.iloc[:, -1]

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

#PCA
def principal_components_analysis(data, n_components):
    # import data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    target = data.ix[:, num_features]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))
    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(features)

    # Model transformation
    new_feature_vector = pca.transform(features)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance sum: ' + str(sum(pca.explained_variance_ratio_)))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    print('\n\n')

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data


if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/test.csv")
    #show_data_info(data)


    #PRIMERA ITERACION
    replace_missing_values_with_constant(data['build_year'], "-1")
    replace_missing_values_with_mode(data, ['life_sq', 'floor', 'max_floor', 'kitch_sq', 'state', 'num_room', 'material', 'railroad_station_walk_km', 'metro_min_walk',
                                            'hospital_beds_raion', 'metro_km_walk'])
    replace_mv_with_constant(data, -1)
    nominal_to_numeric(data)
    convert_data_to_numeric(data)
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
    attribute_subset_selection_with_trees(data, "Regression")
    min_max_scaler(data)

    #SEGUNDA ITERACION
    # replace_missing_values_with_constant(data['build_year'], "-1")
    # replace_missing_values_with_mode(data,
    #                                  ['life_sq', 'floor', 'max_floor', 'kitch_sq', 'state', 'num_room', 'material',
    #                                   'railroad_station_walk_km', 'metro_min_walk',
    #                                   'hospital_beds_raion', 'metro_km_walk'])
    # replace_mv_with_constant(data, -1)
    # nominal_to_numeric(data)
    # convert_data_to_numeric(data)
    #remove_outliers(data, 'full_sq', 5000)
    #remove_outliers(data, 'life_sq', 7000)
    #remove_outliers(data, 'floor', 70)
    #remove_outliers(data, 'max_floor', 100)
    #remove_outliers(data, 'kitch_sq', 1700)
    #remove_outliers(data, 'school_km', 50)
    #remove_outliers(data, 'life_sq', 40)
    #remove_outliers(data, 'industrial_km', 20)
    #remove_outliers(data, 'school_km', 40)
    #remove_outliers(data, 'mosque_km', 40)
    #principal_components_analysis(data, 80)
    #min_max_scaler(data)


    print(data)
    write_file(data, "/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/salida.csv")
    #output.to_csv('out.csv', index=False)