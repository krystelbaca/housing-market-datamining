import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# test['loss'] = np.nan
joined = pd.concat([train])


def feat_imp(model, n_features):
    d = dict(zip(train.columns, model.feature_importances))
    ss = sorted(d, key=d.get, reverse=True)
    top_names = ss[0:n_features]

    plt.figure(figsize=(15, 15))
    plt.title("Feature importances")
    plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
    plt.xlim(-1, n_features)
    plt.xticks(range(n_features), top_names, rotation='vertical')
    # plot the important features #
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()

if __name__ == '__main__':
    #
    # for f in train.columns:
    #     if train[f].dtype == 'object':
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(train[f].values))
    #         train[f] = lbl.transform(list(train[f].values))
    #
    # train_y = train.price_doc.values
    # train_X = train.drop(["id", "timestamp", "price_doc"], axis=1)
    #
    #
    # xgb_params = {
    #     'eta': 0.05,
    #     'max_depth': 8,
    #     'subsample': 0.7,
    #     'colsample_bytree': 0.7,
    #     'objective': 'reg:linear',
    #     'eval_metric': 'rmse',
    #     'silent': 1
    # }
    # train = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    # model = xgb.train(dict(xgb_params, silent=0), train, num_boost_round=100)
    # feat_imp(model, 10)
    train_df = pd.read_csv("train.csv", parse_dates=['timestamp'])
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    dtype_df.groupby("Column Type").aggregate('count').reset_index()

    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count'] > 0]
    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    # plt.show()
    for f in train_df.columns:
        if train_df[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))

    train_y = train_df.price_doc.values
    train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # plot the important features #
    # fig, ax = plt.subplots(figsize=(12, 18))
    # xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    # plt.show()

