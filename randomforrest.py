from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_module import pre_process
import tensorflow as tf


def get_dataset(path, test_ratio):
    raw_data = pd.read_csv(path)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    x_train, x_val, y_train, y_val,x_test = pre_process(x_data, y_data, test_ratio)

    return x_train, x_val, y_train, y_val


def get_randomized_cv(estimator, param_distribution, trails, k_folds):
    randomized_cv = RandomizedSearchCV(estimator=estimator, param_distributions=param_distribution
                                       , n_iter=trails, cv=k_folds, verbose=2, n_jobs=8,
                                       scoring="neg_root_mean_squared_error")

    return randomized_cv


def get_param_grid():
    # number of decision trees
    n_estimators = [int(x) for x in np.linspace(start=200, stop=900, num=7)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # maximum number of levels in trees
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # minimum number of samples required at each leaf node
    min_sample_leaf = [1, 2, 4]

    # method of selecting samples for training each tree
    bootstrap = [True, False]

    # minimum impurity for to split a node
    min_impurity_split = [0.1,0.25,0.40,0.55,0.70,0.99]

    ccp_alpha = [0.1,0.3,0.4,0.6]

    # criterion

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_sample_leaf,
        'bootstrap': bootstrap
    }

    print(random_grid)
    return random_grid


def main():

    path = "./data/train.csv"

    # randomized CV
    trails = 100
    k_folds = 3

    x_train, x_val, y_train, y_val = get_dataset(path, 0.1)

    param_dist = get_param_grid()
    random_cv = get_randomized_cv(RandomForestRegressor(), param_dist, trails, k_folds)

    random_cv.fit(x_train, y_train)

    best_params = random_cv.best_params_

    print(best_params)


if __name__ == "__main__":
    main()
