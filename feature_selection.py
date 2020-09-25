import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import xgboost


def constant_filter(x_data):
    print(f"num cols before constant filter {x_data.columns.shape[0]}")
    selector = VarianceThreshold(threshold=0)
    selector.fit(x_data)
    features_to_keep = x_data.columns[selector.get_support()]
    features_to_drop = [x for x in x_data.columns if x not in features_to_keep]
    x_data.drop(labels=features_to_drop, axis=1, inplace=True)
    print(f"num cols after constant filter {x_data.columns.shape[0]}")
    return x_data


def quasi_constant_filter(x_data, threshold):
    print(f"num cols before quasi constant filter {x_data.columns.shape[0]}")
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(x_data)
    features_to_keep = x_data.columns[selector.get_support()]
    features_to_drop = [x for x in x_data.columns if x not in features_to_keep]
    x_data.drop(labels=features_to_drop, axis=1, inplace=True)
    print(f"num cols after quasi constant filter {x_data.columns.shape[0]}")
    return x_data


def correlated_filter(x_data, correlation_threshold):
    print(f"num cols before correlation filter {x_data.columns.shape[0]}")
    correlated_cols = set()
    correlation_matrix = x_data.corr()

    for colx in range(x_data.columns.shape[0]):
        for coly in range(colx):
            if abs(correlation_matrix.iloc[colx, coly]) > correlation_threshold:
                correlated_cols.add(x_data.columns[coly])

    cols_to_drop = correlated_cols

    x_data.drop(labels=cols_to_drop, axis=1, inplace=True)
    print(f"num cols after correlation filter {x_data.columns.shape[0]}")
    return x_data


def forward_feature_selection(x_data, y_data, n_select):
    n_select = int(n_select)
    print(f"num cols before forward feature selection {x_data.shape[1]}")

    sffs = SFS(RandomForestRegressor(n_estimators=100, n_jobs=4), k_features="best",
               forward=True, floating=False, verbose=2, scoring="neg_mean_squared_log_error", cv=3)
    selected_feat = sffs.fit(x_data, y_data)
    idx = list(selected_feat.k_feature_idx_)
    cols_to_keep = x_data.columns[idx]
    cols_to_drop = [x for x in x_data.columns if x not in cols_to_keep]
    x_data.drop(labels=cols_to_drop, axis=1, inplace=True)
    print(f"num cols after forward feature selection {x_data.shape[1]}")
    return x_data


def forward_feature_selection_xgb(x_data, y_data, n_select):
    n_select = int(n_select)
    print(f"num cols before forward feature selection (xgb) {x_data.shape[1]}")

    params = {'subsample': 0.6, 'objective': 'reg:squarederror',
              'n_estimators': 100, 'min_child_weight': 0.8,
              'max_depth': 10, 'learning_rate': 0.1, 'gamma': 1,
              'alpha': 10}
    xgb_regressor = xgboost.XGBRegressor(objective=params["objective"], min_child_weight=params["min_child_weight"],
                                         learning_rate=params["learning_rate"],
                                         max_depth=params["max_depth"], alpha=params["alpha"], gamma=params["gamma"],
                                         subsample=params["subsample"], n_estimators=params["n_estimators"])

    sffs = SFS(xgb_regressor, k_features="best",
               forward=True, floating=False, verbose=2, scoring="neg_root_mean_squared_error", cv=3)
    print(y_data)
    selected_feat = sffs.fit(x_data, y_data)
    idx = list(selected_feat.k_feature_idx_)
    cols_to_keep = x_data.columns[idx]
    cols_to_drop = [x for x in x_data.columns if x not in cols_to_keep]
    x_data.drop(labels=cols_to_drop, axis=1, inplace=True)
    print(f"num cols after forward feature selection {x_data.shape[1]}")
    return x_data


def tree_derived_feature_importance(x_data, y_data,select_n):
    select_n = int(select_n)
    print(f"number of cols before tree derived feature importance {x_data.columns.shape[0]}")
    select_ = RFE(RandomForestRegressor(n_estimators=100), n_features_to_select=select_n, verbose=2)
    select_.fit(x_data, y_data)
    idx = select_.get_support()
    cols_to_keep = x_data.columns[idx]
    cols_to_drop = [x for x in x_data.columns if x not in cols_to_keep]
    x_data.drop(labels=cols_to_drop, axis=1, inplace=True)
    print(f"number of cols after tree derived feature importance {x_data.columns.shape[0]}")
    return x_data
