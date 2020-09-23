import pandas as pd
import numpy as np


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold,chi2,mutual_info_regression,f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression,Lasso,LinearRegression
from sklearn.tree import DecisionTreeRegressor



from sklearn.feature_selection import SelectKBest,SelectPercentile
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import RFE


from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error


categorical_features = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
                        "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType",
                        "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                        "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                        "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu",
                        "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence",
                        "MiscFeature", "SaleType", "SaleCondition"
                        ]
numerical_features = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1",
                      "BsmtFinSF2",
                      "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
                      "BsmtFullBath",
                      "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                      "Fireplaces",
                      "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                      "3SsnPorch",
                      "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]

not_impute_cat = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
                  "GarageType",
                  "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

def find_missing(data):
    missing_cat_vals_features = []
    missing_num_vals_features = []
    for feature in categorical_features:
        data_series = data[feature]
        missing_val = data_series.isnull()

        if missing_val.sum() != 0:
            missing_cat_vals_features.append(feature)

    for feature in numerical_features:
        data_series = data[feature]
        missing_val = data_series.isnull()
        if missing_val.sum() != 0:
            missing_num_vals_features.append(feature)

    return missing_cat_vals_features, missing_num_vals_features

def fill_missing(feature,type,data):
    if type == "cat":
        imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
    elif type == "num":
        imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

    return imputer.fit_transform(data[feature].values.reshape(-1, 1))


def filter_constant_numeric_features(x_data):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(x_data)
    features_to_keep = x_data.columns[constant_filter.get_support()]
    return x_data[features_to_keep]


def filter_constant_cat_features(x_data):
    return [x for x in x_data.columns if len(x_data[x].unique() == 1)]


def filter_quasi_constant_numeric_features(x_data,threshold):
    quasi_filter = VarianceThreshold(threshold=threshold)

    quasi_filter.fit(x_data)
    columns_to_keep = x_data.columns[quasi_filter.get_support()]

    return x_data[columns_to_keep]


def filter_quasi_constant_cat_features(x_data,threshold):
    quasi_constant_features = []
    for column in x_data.columns:
        data_series = x_data[column]
        dominant = (data_series.value_counts()/np.float(x_data.shape[0])).sort_values(ascending=False).values[0]

        if dominant > threshold:
            quasi_constant_features.append(column)

    features_to_keep = [x for x in x_data.columns if x not in quasi_constant_features]

    return x_data[features_to_keep]


def filter_correlated_features(data,corr_thresh):
    corr_cols = set()
    corr_matrix = data.corr()

    for col in range(corr_matrix.columns.shape[0]):
        for col2 in range(col):
            if abs(corr_matrix.iloc[col, col2]) > corr_thresh:
                col_name = data.columns[col2]
                corr_cols.add(col_name)

    data.drop(labels=corr_cols, axis=1, inplace=True)
    return data


def filter_mutual_info(x_data,y_data,select_no):

    print("Applying mutual info filter only on numeric data")
    num_cols = x_data.select_dtypes(include="number").columns
    print(f"Number of numeric cols before MI filtering {x_data.select_dtypes(include='number').columns.shape[0]}")
    print(f"Number of cat cols before MI filtering {x_data.select_dtypes(include='object').columns.shape[0]}")
    temp_data = x_data[num_cols]
    filter_ = SelectKBest(mutual_info_regression,select_no).fit(temp_data,y_data)
    cols_to_keep = temp_data.columns[filter_.get_support()]
    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]

    x_data = x_data.drop(labels= cols_to_drop,axis=1,inplace = False)
    print(f"Number of numeric cols after MI filtering {x_data.select_dtypes(include='number').columns.shape[0]}")
    print(f"Number of cat cols after MI filtering {x_data.select_dtypes(include='object').columns.shape[0]}")

    return x_data

def filter_chi_fischer(x_data,selec_no): # only for categorical and binary classification
    return 0


def filter_univariate_anova(x_data,y_data,select_no):

    print(f"Applying Ftest only on numeric variables")
    print(f"cat variables before f-test {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables before f-test {x_data.select_dtypes(include='number').columns.shape}")

    num_cols = x_data.select_dtypes(include='number').columns

    temp = x_data[num_cols]

    ftest_filter = SelectKBest(f_regression,select_no).fit(temp,y_data)

    cols_to_keep =  num_cols[ftest_filter.get_support()]

    cols_to_drop = [x for x in num_cols if x not in cols_to_keep ]

    print(f"droping cols {cols_to_drop.__len__()}")

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)

    print(f"cat variables after f-test {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables after f-test {x_data.select_dtypes(include='number').columns.shape}")

    return x_data

def filter_univariate_mse(x_data,y_data,x_test,y_test,select_no):
    print("Applying Univariate MSE to numerical data")
    print(f"cat variables before univariate mse filtering {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables before univariate mse filtering {x_data.select_dtypes(include='number').columns.shape}")

    num_cols = x_data.select_dtypes(include='number').columns

    temp = x_data[num_cols]

    roc_values = []

    for feature in num_cols:
        dtr = DecisionTreeRegressor()
        dtr.fit(temp[feature].values.reshape((-1,1)),y_data.values.reshape((-1,1)))
        predictions = dtr.predict(x_test[feature].values.reshape((-1,1)))
        score = mean_squared_error(y_test,predictions)

        roc_values.append(score)
    scores = pd.Series(roc_values)
    scores.index = num_cols
    scores = scores.sort_values(axis=0,ascending=True)

    cols_to_keep = scores.iloc[0:select_no]
    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)

    print(f"cat variables after univariate mse filtering {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables after univariate mse filtering {x_data.select_dtypes(include='number').columns.shape}")
    return x_data

def forward_feature_selection(x_data,y_data,n_select):
    print("Applying forward feature selection to numerical data")
    print(f"cat variables before forward feature selection {x_data.select_dtypes(include='object').shape}")
    print(f"numeric variables before forward feature selection {x_data.select_dtypes(include='number').shape}")
    num_cols = x_data.select_dtypes(include='number').columns
    temp = x_data[num_cols]
    sfsf = sfs(RandomForestRegressor(n_jobs=5),k_features=n_select,forward=True,floating=False,verbose=2,cv=3,scoring='r2')
    sfsf.fit(temp,y_data)
    idx = sfsf.k_feature_idx_
    idx = list(idx)
    cols_to_keep = num_cols[idx]

    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]
    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)

    print(f"cat variables after forward feature selection {x_data.select_dtypes(include='object').columns}")
    print(f"numeric variables after forward feature selection {x_data.select_dtypes(include='number').columns}")
    return x_data


def backward_feature__selection(x_data,y_data,n_select):
    print("Applying backward feature to numerical data")
    print(f"cat variables before backward feature selection {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables before backward feature selection {x_data.select_dtypes(include='number').columns.shape}")

    numeric_cols = x_data.select_dtypes(include="number").columns

    temp = x_data[numeric_cols]
    bfsf = sfs(RandomForestRegressor(n_jobs=5),k_features=n_select,forward=False,verbose=2,floating=False,cv=3,scoring='r2')
    bfsf.fit(temp,y_data)

    idx = bfsf.k_feature_idx_
    idx = list(idx)

    cols_to__keep = x_data.columns[idx]
    cols_to_drop = [x for x in numeric_cols if x not in cols_to__keep]

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)
    print(f"cat variables after backward feature selection {x_data.select_dtypes(include='object').columns}")
    print(f"numeric variables after backward feature selection {x_data.select_dtypes(include='number').columns}")

    return x_data

def exhaustive_feature_selection(x_data,y_data,min_feat,max_feat):
    print(f"Applying exhaustive feature selection to numeric data")
    print(f"cat variables before backward feature selection {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables before backward feature selection {x_data.select_dtypes(include='number').columns.shape}")

    numeric_cols = x_data.select_dtypes(include='number').columns

    temp = x_data[numeric_cols]

    efs = EFS(RandomForestRegressor(n_jobs=4),max_features=max_feat,min_features=min_feat,scoring='r2',print_progress=True,cv=2)

    efs.fit(temp,y_data)

    idx = efs.best_idx_

    print(idx)

    idx = list(idx)

    cols_to_keep = x_data.columns[idx]
    cols_to_drop = [x for x in numeric_cols if x not in cols_to_keep]

    print(cols_to_drop.__len__())

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)
    print(f"cat variables after exhaustive feature selection {x_data.select_dtypes(include='object').columns}")
    print(f"numeric variables after exhaustive  feature selection {x_data.select_dtypes(include='number').columns}")
    return x_data


def lasso_selection(x_data,y_data,penalty):
    print("Applying LASSO regularization to numeric scaled features")
    print(f"cat variables before lasso regularization selection {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variable before lass regularization selection{x_data.select_dtypes(include='number').columns.shape}")
    temp = x_data.select_dtypes(include='number')

    standard_scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
    standard_scalar.fit(temp)
    standard_scalar.transform(temp)
    lasso_regularization = SelectFromModel(Lasso(alpha=2000))
    # (1 / (2 * n_samples)) * | | y - Xw | | ^ 2_2 + alpha * | | w | | _1
    lasso_regularization.fit(temp,y_data)
    cols_to_keep = temp.columns[lasso_regularization.get_support()]
    cols_to_drop = [x for x in temp.columns if x not in cols_to_keep]
    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)
    print(f"cat variables after lasso regularization selection {x_data.select_dtypes(include='object').columns.shape}")
    print(f"numeric variables after lassa regularization selection{x_data.select_dtypes(include='number').columns.shape}")

    return x_data

def linear_coefficients_filtering(x_data,y_data):
    print(f"Applying linear coefficients based filtering on scaled numeric variables")
    print(f"cat variables before linear coefficients based filtering {x_data.select_dtypes(include='object').shape}")
    print(f"var variables before linear coefficients based filtering {x_data.select_dtypes(include='number').shape}")

    num_cols = x_data.select_dtypes(include='number').columns
    temp = x_data[num_cols]

    scalar = StandardScaler(with_mean=True,with_std=True)
    temp = scalar.fit_transform(temp)
    temp = pd.DataFrame(data=temp, columns=num_cols)

    linear_coff_filter = SelectFromModel(estimator=LinearRegression())
    linear_coff_filter.fit(temp,y_data)

    param_coff = linear_coff_filter.estimator_.coef_
    param_coff = pd.Series(param_coff)
    param_coff.index = temp.columns

    idx = linear_coff_filter.get_support()

    cols_to_keep = num_cols[idx]

    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]

    print(cols_to_keep)
    print(cols_to_drop)

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)
    print(f"cat variables after linear coefficients based filtering {x_data.select_dtypes(include='object').shape}")
    print(f"var variables after linear coefficients based filtering {x_data.select_dtypes(include='number').shape}")
    return x_data


def select_from_tree(x_data,y_data,select_k):
    print(f"Applying tree derived importance filter")
    print(f"cat variables before tree derived importance filter  {x_data.select_dtypes(include='object').shape}")
    print(f"num variables before tree derived importance filter  {x_data.select_dtypes(include='number').shape}")

    num_cols = x_data.select_dtypes(include='number').columns

    temp = x_data[num_cols]

    select_ = SelectFromModel(estimator=RandomForestRegressor(n_estimators=100))
    select_.fit(temp,y_data)

    cols_to_keep = temp.columns[select_.get_support()]
    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)

    print(f"cat variables after tree derived importance filter  {x_data.select_dtypes(include='object').shape}")
    print(f"num variables after tree derived importance filter  {x_data.select_dtypes(include='number').shape}")

    return x_data


def select_from_tree_recursively(x_data,y_data,select_k):
    print(f"Applying tree derived importance filter")
    print(f"cat variables before tree derived recursive importance filter  {x_data.select_dtypes(include='object').shape}")
    print(f"num variables before tree derived recursive importance filter  {x_data.select_dtypes(include='number').shape}")

    num_cols = x_data.select_dtypes(include='number').columns

    temp = x_data[num_cols]

    select_ = RFE(estimator=RandomForestRegressor(n_estimators=100),n_features_to_select=10)
    select_.fit(temp,y_data)

    cols_to_keep = temp.columns[select_.get_support()]
    cols_to_drop = [x for x in num_cols if x not in cols_to_keep]

    x_data.drop(labels=cols_to_drop,axis=1,inplace=True)

    print(f"cat variables after tree derived recursive importance filter  {x_data.select_dtypes(include='object').shape}")
    print(f"num variables after tree derived recursive importance filter  {x_data.select_dtypes(include='number').shape}")

    return x_data


def pre_process(data,test_ratio,variance_threshold,corr_threshold,select_k_mi,select_k_ftest,select_k_mse,select_k_sfs,select_k_sbs,min_feat,max_feat):

    missing_cat_feature, missing_num_features = find_missing(data)

    print(f"Missing Cat: {missing_cat_feature.__len__()}")
    print(f"Missing Num: {missing_num_features.__len__()}")

    #print(not_impute_cat.__len__())

    missing_cat_feature = [feat for feat in missing_cat_feature if feat not in not_impute_cat]

    for nan_feat in missing_cat_feature:
        data[nan_feat] = fill_missing(nan_feat,"cat",data)

    for nan_feat in missing_num_features:
        data[nan_feat] = fill_missing(nan_feat, "num",data)

    missing_cat_feature, missing_num_features = find_missing(data)

    print(f"Missing Cat: {missing_cat_feature.__len__()}")
    print(f"Missing Num: {missing_num_features.__len__()}")

    x_data = data.iloc[:, 1:-1]
    print(x_data.shape)
    y_data = data["SalePrice"]

    x_train,x_val,y_train,y_val =  train_test_split(x_data, y_data, test_size=test_ratio, random_state=0)

    print(f"before filtering {x_data.shape}")

    # filter constant numeric
    x_train_nums = filter_constant_numeric_features(x_train[numerical_features])
    print(x_train_nums.shape)

    # filter constant categorical
    x_train_cats = x_train[filter_constant_cat_features(x_train[categorical_features])]
    print(x_train_cats.shape)

    x_data = pd.concat([x_train_nums,x_train_cats],axis=1)
    print(f"after constant filtering {x_data.shape}")

    # filter quasi constant numeric
    x_train_nums = filter_quasi_constant_numeric_features(x_data[numerical_features],variance_threshold)
    print(x_train_nums.shape)

    # filter quasi constant categorical
    x_train_cats = filter_quasi_constant_cat_features(x_data[categorical_features],variance_threshold)
    print(x_train_cats.shape)

    x_data = pd.concat([x_train_nums, x_train_cats], axis=1)
    print(f"after quasi constant filtering {x_data.shape}")

    # filter high correlated features (only numeric types)
    x_data = filter_correlated_features(x_data,corr_threshold)
    print(f"after high correlated filtering {x_data.shape}")

    # filter based on mutual info ranking
    x_data = filter_mutual_info(x_data,y_train,select_k_mi)
    print(f"after high mutual info filtering {x_data.shape}")




    # filter based on chi-fischer values ranking
    # not applicable

    # filter based on F-test values ranking
    x_data = filter_univariate_anova(x_data,y_train,select_k_ftest)
    print(f"after ftest value ranking filtering {x_data.shape}")

    # filter based on univariate mse values ranking
    x_data = filter_univariate_mse(x_data,y_train,x_val,y_val,select_k_mse)
    print(f"after univariate mse ranking filtering {x_data.shape}")

    # filter based on forward feature selection
    # x_data = forward_feature_selection(x_data, y_train,select_k_sfs)
    # print(f"after forward feature selection {x_data.shape}")

    # filter based on backward feature selection
    # x_data = backward_feature__selection(x_data,y_train,select_k_sbs)
    # print(f"after sequential backward feature selection {x_data.shape}")

    # filter based on exhaustive feature selection
    # x_data = exhaustive_feature_selection(x_data,y_train,min_feat,max_feat)
    # print(f"after exhaustive  feature selection {x_data.shape}")

    # filter based on LASSO regularization
    x_data = lasso_selection(x_data,y_train,1)
    print(f"after lasso regularization selection {x_data.shape}")

    # filter based on linear coefficients

    linear_coefficients_filtering(x_data,y_train)
    print(f"after linear coefficient  selection {x_data.shape}")

    print("===========================================")
    # filter bases on tree derived importance
    select_from_tree(x_data, y_train, 2)
    print(f"after tree derived importance  selection {x_data.shape}")
    print("===========================================")

def main():
    raw_data = pd.read_csv("./data/train.csv")
    print(raw_data.columns.shape)

    pre_process(raw_data,0.1,variance_threshold=0.99,corr_threshold=0.8,select_k_mi=20,select_k_ftest=15,select_k_mse=12,select_k_sfs=10,select_k_sbs=10,min_feat=5,max_feat=9)



if __name__ == "__main__":
    main()