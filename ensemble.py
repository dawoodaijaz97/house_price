from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_module import pre_process
import tensorflow as tf
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from matplotlib import pyplot


def get_dataset(val_ratio,for_model):
    path = "./data/train.csv"
    path2 = "./data/test.csv"

    raw_data1 = pd.read_csv(path)
    print(raw_data1.shape)
    raw_data2 = pd.read_csv(path2)
    print(raw_data2.shape)
    raw_data = pd.concat((raw_data1,raw_data2),axis=0)
    print(f"Raw data {raw_data.shape}")

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data1.iloc[:, -1]

    x_train, x_val, y_train, y_val,x_test = pre_process(x_data, y_data, val_ratio,for_model)
    return x_train, x_val, y_train, y_val,x_test


def get_best_xgb_model(params):
    xgb_regressor = xgboost.XGBRegressor(objective=params["objective"], min_child_weight=params["min_child_weight"],
                                         learning_rate=params["learning_rate"],
                                         max_depth=params["max_depth"], alpha=params["alpha"], gamma=params["gamma"],
                                         subsample=params["subsample"], n_estimators=params["n_estimators"])

    return xgb_regressor


def get_best_rf_model(params):
    optimal_model = RandomForestRegressor(n_estimators=params["n_estimators"],
                                          min_samples_split=params["min_samples_split"]
                                          , min_samples_leaf=params["min_samples_leaf"],
                                          max_features=params["max_features"],
                                          max_depth=params["max_depth"], ccp_alpha=params["ccp_alpha"],
                                          min_impurity_split=params["min_impurity_split"],
                                          bootstrap=params["bootstrap"],
                                          criterion=params["criterion"])
    return optimal_model

def get_best_lr_model(params):

    return Ridge(alpha=0.1)


def train_model(x_train, y_train, model):
    model.fit(x_train, y_train)
    return model, model.predict(x_train)


def eval_model(x_val, y_val, model):
    preds = model.predict(x_val)
    msle = tf.keras.metrics.mean_squared_logarithmic_error(y_val, preds)
    rmsle = tf.math.sqrt(msle)
    print(rmsle)
    return preds

def get_preds(x_test,model):
    preds = model.predict(x_test)
    return preds


def create_ensemble_data(y_train_pred1, y_train_pred2, y_val_pred1, y_val_pred2):
    y_train_pred1 = np.reshape(y_train_pred1, (-1, 1))
    y_train_pred2 = np.reshape(y_train_pred2, (-1, 1))

    y_val_pred1 = np.reshape(y_val_pred1, (-1, 1))
    y_val_pred2 = np.reshape(y_val_pred2, (-1, 1))

    x_train = np.concatenate((y_train_pred1, y_train_pred2), axis=1)
    x_val = np.concatenate((y_val_pred1, y_val_pred2), axis=1)

    return x_train, x_val


def main():
    x_train_xg, x_val_xg, y_train, y_val,x_test_xg = get_dataset(0.1,"xg")
    x_train_rf, x_val_rf, y_train, y_val,x_test_rf = get_dataset(0.1,"rf")
    x_train_lr, x_val_lr, y_train, y_val,x_test_lr = get_dataset(0.1,"xg")

    best_params_xg_model = {'subsample': 0.6, 'objective': 'reg:squarederror',
                            'n_estimators': 666, 'min_child_weight': 0.8,
                            'max_depth': 10, 'learning_rate': 0.1, 'gamma': 1,
                            'alpha': 10}
    xgb_regressor = get_best_xgb_model(best_params_xg_model)

    best_params_rf = {'n_estimators': 666, 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'max_features': 'sqrt', 'max_depth': 20, 'ccp_alpha': 0.6,
                      'min_impurity_split': 0.7,
                      'bootstrap': False, 'criterion': 'mse'}
    rf_regressor = get_best_rf_model(best_params_rf)

    lr_regressor = get_best_lr_model(None)

    xgb_regressor, y_train_pred1 = train_model(x_train_xg, y_train, xgb_regressor)
    rf_regressor, y_train_pred2 = train_model(x_train_rf, y_train, rf_regressor)
    lr_regressor,y_train_pred3 = train_model(x_train_lr,y_train,lr_regressor)

    y_val_pred1 = eval_model(x_val_xg, y_val, xgb_regressor)
    y_val_pred2 = eval_model(x_val_rf, y_val, rf_regressor)
    y_val_pred3 = eval_model(x_val_lr, y_val, lr_regressor)

    x_train, x_val = create_ensemble_data(y_train_pred1, y_train_pred2, y_val_pred1, y_val_pred2)

    linear_model = LinearRegression()
    linear_model, y_g = train_model(x_train, y_train, linear_model)

    eval_model(x_val, y_val, linear_model)

    y_test_pred1 = get_preds(x_test_xg,xgb_regressor)
    y_test_pred2 = get_preds(x_test_rf,rf_regressor)

    y_test_pred1 = np.reshape(y_test_pred1, (-1, 1))
    y_test_pred2 = np.reshape(y_test_pred2, (-1, 1))
    y_test_pred = np.concatenate((y_test_pred1, y_test_pred2), axis=1)

    y_test = get_preds(y_test_pred,linear_model)

    y_test = pd.Series(y_test)
    print(f"y test shape {y_test.shape}")
    y_test.index = pd.RangeIndex(start=1461, stop=2920, step=1)

    y_test.to_csv("./submission.csv",sep=",")

    print(y_test)


if __name__ == "__main__":
    main()
