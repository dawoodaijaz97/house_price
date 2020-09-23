from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_module import pre_process
import tensorflow as tf


def get_dataset(path, eval_ratio):
    raw_data = pd.read_csv(path)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    x_train, x_val, y_train, y_val,x_test = pre_process(x_data, y_data, 0.1)
    print(x_test.shape)
    if eval_ratio == 0:
        x_data = np.concatenate((x_train, x_val), axis=0)
        y_data = np.concatenate((y_train, y_val), axis=0)
        return x_data, y_data

    return x_train, x_val, y_train, y_val


def get_randomized_cv(estimator, param_distribution, trails, k_folds):
    randomized_cv = RandomizedSearchCV(estimator=estimator, param_distributions=param_distribution
                                       , n_iter=trails, cv=k_folds, verbose=2, n_jobs=8,
                                       scoring="neg_root_mean_squared_error")

    return randomized_cv


def get_param_grid():
    random_grid = {
        'objective': ["reg:squarederror"],
        'learning_rate': [0.1],
        'max_depth': [10, 20, 30],
        'alpha': [5, 10, 15],
        'min_child_weight': [0.8, 1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.5, 0.6, 0.8, 1.0],
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=900, num=7)]
    }

    print(random_grid)
    return random_grid


def main():
    path = "./data/train.csv"

    k_folds = 5
    trials = 100

    x_data, y_data = get_dataset(path, 0.0)

    param_dist = get_param_grid()

    xgb_regressor = xgboost.XGBRegressor()

    random_cv_search = RandomizedSearchCV(xgb_regressor, param_dist, n_iter=trials, n_jobs=8, cv=k_folds, verbose=0)
    random_cv_search.fit(x_data, y_data)

    best_params = random_cv_search.best_params_
    print(best_params)


if __name__ == "__main__":
    main()
