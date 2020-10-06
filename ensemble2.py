from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_module import pre_process
import tensorflow as tf
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from matplotlib import pyplot
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor


def get_dataset(val_ratio,for_model,is_duan):
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

    x_train, x_val, y_train, y_val,x_test = pre_process(x_data, y_data, val_ratio,for_model,is_duan)
    return x_train, x_val, y_train, y_val,x_test


def get_best_lr_model(params):

    return Ridge(alpha=0.1,random_state=0)

def get_best_xg_model(params):
    xg_regressor = XGBRegressor(learning_rate=0.01,
                                n_estimators=6000,
                                max_depth=4,
                                min_child_weight=0,
                                gamma=0.6,
                                subsample=0.7,
                                colsample_bytree=0.7,
                                objective='reg:squarederror',
                                nthread=-1,
                                scale_pos_weight=1,
                                seed=27,
                                reg_alpha=0.00006,
                                random_state=42)
    return xg_regressor


def get_best_rf_model(params):
    rf = RandomForestRegressor(n_estimators=1200,
                               max_depth=15,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               max_features=None,
                               oob_score=True,
                               random_state=42)
    return rf


def get_best_boost_model(params):
    gbr = GradientBoostingRegressor(n_estimators=6000,
                                    learning_rate=0.01,
                                    max_depth=4,
                                    max_features='sqrt',
                                    min_samples_leaf=15,
                                    min_samples_split=10,
                                    loss='huber',
                                    random_state=42)
    return gbr



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


def get_mean_exp_residual(y_preds,y_trues):
    sum_exp_abs_diff = 0
    residual_vector = []
    for i in range(y_preds.shape[0]):
        abs_diff = (tf.math.subtract(y_preds[i],y_trues.iloc[i]))
        residual_vector.append(abs_diff)
        exp_abs_diff = tf.math.exp(abs_diff)
        sum_exp_abs_diff = sum_exp_abs_diff + exp_abs_diff
    print(f"total sum {sum_exp_abs_diff}")
    mean_exp_sum_abs_diff = sum_exp_abs_diff.numpy()/y_preds.shape[0]
    print(mean_exp_sum_abs_diff)
    return mean_exp_sum_abs_diff,residual_vector

def dist_prob_plot2(data):
    plt.subplots(figsize=(12,9))
    sns.distplot(data,kde=True)
    plt.show()
    stats.probplot(data,plot=plt)
    plt.show()


def main():

    x_train, x_val, y_train, y_val,x_test= get_dataset(0.1,"xg",is_duan=False)

    x_train_2,x_val_2,y_train,y_val,x_test_2 = get_dataset(0.1,"rf",is_duan=False)

    lr_regressor = get_best_lr_model(None)
    lr_regressor,y_train_pred_1 = train_model(x_train,y_train,lr_regressor)

    xg_regressor = get_best_xg_model(None)
    xg_regressor, y_train_pred_2 = train_model(x_train, y_train, xg_regressor)

    gb_regressor = get_best_xg_model(None)
    gb_regressor,y_train_pred_3 = train_model(x_train,y_train,gb_regressor)

    rf_regressor = get_best_rf_model(None)
    rf_regressor, y_train_pred_4 = train_model(x_train_2, y_train, rf_regressor)

    stack_gen = StackingCVRegressor(regressors=(xg_regressor,lr_regressor,gb_regressor,rf_regressor),
                                    meta_regressor=xg_regressor,
                                    use_features_in_secondary=True)

    stack_gen, y_train_pred_5= train_model(x_train, y_train, stack_gen)

    eval_model(x_val, y_val, lr_regressor)
    eval_model(x_val, y_val, xg_regressor)
    eval_model(x_val, y_val, gb_regressor)
    eval_model(x_val_2, y_val, rf_regressor)
    eval_model(x_val,y_val,stack_gen)

    expected_exp_residual,residual_vector = get_mean_exp_residual(y_train_pred_5,y_train)

    print(expected_exp_residual)

    def blended_predictions(x1,x2):
        return (
                (0.35 * lr_regressor.predict(x1))
                (0.1 * gb_regressor.predict(x1)) +
                (0.1 * xg_regressor.predict(x1)) +
                (0.05 * rf_regressor.predict(x2)) +
                (0.1 * stack_gen.predict(np.array(x1))))

    y_test = blended_predictions(x_test,x_test_2)

    print(y_test)

    y_test = tf.math.exp(y_test)
    y_test = tf.math.multiply(y_test,expected_exp_residual)
    y_test = y_test.numpy()
    print(y_test)
    y_test = pd.Series(y_test)
    y_test.index = pd.RangeIndex(start=1461, stop=2920, step=1)
    y_test.to_csv("./data/submission.csv",sep=",")


if __name__ == "__main__":
    main()
