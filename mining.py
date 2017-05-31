"""
This file show the use of Decision Tree function of sklearn library
for more info: http://scikit-learn.org/stable/modules/tree.html
Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import numpy
from sklearn import preprocessing
from utils import utils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import tree

import pydotplus
#Bagging method
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

#Boosting method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

#Random Forest method
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import model_selection

from sklearn import preprocessing
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
    :param data: numpy array
    :return: four subsets that represents data train and data test
    """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test

def decision_tree_training(data):
    """
    This function train and return a decision tree model
    :param data: numpy array
    :return: decision tree model
    """

    print(feature_names)

    data_features = data.iloc[:, 0:-1]
    data_targets = numpy.asarray(data.iloc[:,-1], dtype="int16")

    #Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = data_splitting(data_features, data_targets, 0.25)

    #Model declaration
    """
    Parameters to select:
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """
    dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dec_tree.fit(data_features_train, data_targets_train)

    #Model evaluation
    test_data_predicted = dec_tree.predict(data_features_test)
    score = metrics.accuracy_score(data_targets_test, test_data_predicted)

    print("Model Score: " + str(score))
    print("Probability of each class: \n")
    #Measure probability of each class
    prob_class = dec_tree.predict_proba(data_features_test)
    print(prob_class)

    print("Feature Importance: \n")
    print(dec_tree.feature_importances_)

    # Draw the tree
    dot_data = tree.export_graphviz(dec_tree, out_file = None,
                                         feature_names = feature_names,
                                         class_names = data_targets,
                                         filled=True, rounded=True,
                                         special_characters=False)

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("../data/decision_tree.pdf")

def convert_data_to_numeric(data):
    numpy_data = data.values
    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        if(type(temp[0]).__name__  == 'str'):
            dict = numpy.unique(numpy_data[:, i])
            # print(dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[numpy.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:, i] = temp
    return numpy_data

def nominal_to_numeric(data):
   for f in data.columns:
       if data[f].dtype=='object':
           lbl = preprocessing.LabelEncoder()
           lbl.fit(list(data[f].values.astype('str')))
           data[f] = lbl.transform(list(data[f].values.astype('str')))

def mlp_classifier(data):
    #load data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    targets = data.ix[:, num_features]

    print(features)
    print(targets)

    # Data splitting
    features_train, features_test, targets_train, targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    hidden_layer_sizes: its an array in which each element represents a new layer with "n" neurones on it
            Ex. (3,4) = Neural network with 2 layers: 3 neurons in the first layer and 4 neurons in the second layer
            Ex. (25) = Neural network with one layer and 25 neurons
            Default = Neural network with one layer and 100 neurons
    activation: "identity", "logistic", "tanh" or "relu". Default: "relu"
    solver: "lbfgs", "sgd" or "adam" Default: "adam"
    ###Only used with "sgd":###
    learning_rate_init: Neural network learning rate. Default: 0.001
    learning_rate: Way in which learning rate value change through iterations.
            Values: "constant", "invscaling" or "adaptive"
    momentum: Default: 0.9
    early_stopping: The algorithm automatic stop when the validation score is not improving.
            Values: "True" or "False". Default: False
    """
    neural_net = MLPClassifier(
        hidden_layer_sizes=(25),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(features_train, targets_train.values.ravel())

    # Model evaluation
    test_data_predicted = neural_net.predict(features_test)
    score = metrics.accuracy_score(targets_test, test_data_predicted)

    logger.debug("Model Score: %s", score)

def ensemble_methods_classifiers(train, test):
    # load data
    num_features = len(train.columns) - 1

    features = train.ix[:, 1:num_features]
    targets = train.ix[:, num_features]

    print(features)
    print(targets)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    n_estimators: The number of base estimators in the ensemble.
            Values: Random Forest and Bagging. Default 10
                    AdaBoost. Default: 50
    ###Only for Bagging and Boosting:###
    base_estimator: Base algorithm of the ensemble. Default: DecisionTree
    ###Only for Random Forest:###
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """

    names = ["Bagging Classifier", "AdaBoost Classifier", "Random Forest Classifier", "Decision Tree Regressor", "SVR",
             "KNeighbors Regressor"]

    models = [
        BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        AdaBoostClassifier(
            n_estimators=10,
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        RandomForestClassifier(
            criterion='gini',
            max_depth=10
        ),
        tree.DecisionTreeRegressor(
            criterion='mse'
        ),
        SVR(
            kernel='rbf',
            C=1e3,
            gamma=0.1
        ),
        KNeighborsRegressor()
    ]

    for name, em_clf in zip(names, models):
        logger.info("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train.values.ravel())

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)

        # Get predictions to Kaggle
        kaggle_predictions = em_clf.predict(test.ix[:, 1:num_features])

        # Generate CSV for Kaggle with csv package:
        path = "../resources/predicted_kaggle_" + str(name) + ".csv"

        # Generate CSV for Kaggle with pandas (easiest way)
        df_predicted = pandas.DataFrame({'id': test.ix[:, 0], 'price_doc': kaggle_predictions})

        df_predicted.to_csv(path, index=False)

        error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)
        logger.debug('Total Error: %s', error)

if __name__ == '__main__':
    print("DATA LOADING...")
    data = utils.load_data("salida.csv")

    feature_names = []
    for column in data.columns:
        feature_names.append(column)

    print("DATA CONVERTING...")

    train_data = convert_data_to_numeric(data)
    train_data = nominal_to_numeric(train_data)
    #decision_tree_training(train_data)
    mlp_classifier(data)
    #ensemble_methods_classifiers(train_data, test)