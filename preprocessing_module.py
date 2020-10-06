import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats as sns
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from encoder import encode_vars
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


not_impute_cat = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
                  'GarageYrBlt', 'GarageArea', 'GarageCars'
                  ]

ordinal_vars = ["Street", "Alley", "LotShape", "Utilities", "LandSlope", "ExterQual", "ExterCond",
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir",
                "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", ]

cols_to_select_rf = ['MSSubClass', 'MSZoning', 'Neighborhood', 'HouseStyle', 'OverallQual',
                     'OverallCond', 'YearRemodAdd', 'BsmtExposure', 'BsmtFinType1',
                     'BsmtFinSF1', 'BsmtFinSF2', 'HeatingQC', 'CentralAir', 'TotalBsmtSF',
                     '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                     'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional',
                     'FireplaceQu', 'GarageFinish', 'GarageCond', 'ScreenPorch',
                     'MiscFeature', 'SaleCondition', "GarageCars"]

cols_to_select_xg = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig',
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                     'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofMatl',
                     'MasVnrType', 'ExterCond', 'Foundation', 'BsmtExposure', 'BsmtFinType1',
                     'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'CentralAir',
                     'Electrical', 'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath',
                     'BsmtHalfBath', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
                     'FireplaceQu', 'GarageFinish', 'GarageCars', 'PavedDrive', '3SsnPorch',
                     'ScreenPorch', 'PoolQC', 'Fence', 'YrSold', 'SaleType',
                     'SaleCondition']

high_correlation = ["OverallQual",
                    "GrLivArea",
                    "GarageCars",
                    "TotalBsmtSF",
                    "FullBath",
                    "TotRmsAbvGrd",
                    "YearBuilt",
                    "YearRemodAdd", ]

skewed_cols = ['MiscVal',
               'PoolArea',
               'LotArea',
               '3SsnPorch',
               'LowQualFinSF',
               'KitchenAbvGr',
               'BsmtFinSF2',
               'ScreenPorch',
               'BsmtHalfBath',
               'EnclosedPorch',
               'MasVnrArea',
               'OpenPorchSF',
               'LotFrontage',
               'BsmtFinSF1',
               'WoodDeckSF',
               'MSSubClass',
               '1stFlrSF', ]

discretize = ["GarageType"]


def fill_missing(x_data, strategy):
    if strategy == "mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    elif strategy == "zero":
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    elif strategy == "frequent":
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    return imputer.fit_transform(x_data)


def remove_sale_price_outliers(x_data, y_data):
    print(y_data.shape)
    print(x_data.shape)
    scaler = StandardScaler()
    scaled_y_data = scaler.fit_transform(np.reshape(y_data.values, (-1, 1)))
    scaled_y_data = pd.Series(np.reshape(scaled_y_data, (-1,)))

    scaled_y_data = scaled_y_data.sort_values(ascending=False)

    low_range = scaled_y_data[-5:]
    high_range = scaled_y_data[:5]

    y_data.drop(labels=low_range.index, inplace=True)
    y_data.drop(labels=high_range.index, inplace=True)

    x_test = x_data.iloc[1460:, :]

    x_data = x_data.iloc[:1460, :]
    y_data = y_data.iloc[:1460]

    x_data.drop(labels=low_range.index, inplace=True, axis=0)
    x_data.drop(labels=high_range.index, inplace=True, axis=0)

    x_data = pd.concat((x_data, x_test), axis=0, ignore_index=True)

    print(y_data.shape)
    print(x_data.shape)
    return x_data, y_data


def remove_outliers(x_data, y_data, up, down, col):
    print(x_data.shape)
    x_test = x_data.iloc[1460:, :]

    x_data = x_data.iloc[:1460, :]
    y_data = y_data.iloc[:1460]

    scaler = StandardScaler()
    scaled_x_data = scaler.fit_transform(np.reshape(x_data[col].values, (-1, 1)))
    scaled_x_data = pd.Series(np.reshape(scaled_x_data, (-1,)))

    scaled_x_data = scaled_x_data.sort_values(ascending=False)

    if down != 0:
        low_range = scaled_x_data[-down:]
    else:
        low_range = scaled_x_data[-1:-1]
    if up != 0:
        high_range = scaled_x_data[:up]
    else:
        high_range = scaled_x_data[0:0]

    y_data.drop(labels=low_range.index, inplace=True)
    y_data.drop(labels=high_range.index, inplace=True)

    x_data.drop(labels=low_range.index, inplace=True, axis=0)
    x_data.drop(labels=high_range.index, inplace=True, axis=0)

    x_data = pd.concat((x_data, x_test), axis=0, ignore_index=True)

    print(y_data.shape)
    print(x_data.shape)
    return x_data, y_data


def log_transform_sale_price(data):
    data = np.log(data)
    return pd.Series(data)


def log_transform_zeroed_vals(x_data, has_col, col):
    x_data[has_col] = pd.Series(len(x_data[col]))
    x_data[has_col] = 0
    x_data.loc[x_data[col] > 0, has_col] = 1

    x_data.loc[x_data[has_col] == 1, col] = np.log(x_data[col])
    return x_data


def dist_prob_plot2(data):
    plt.subplots(figsize=(12, 9))
    sns.distplot(data, kde=True)
    plt.show()
    stats.probplot(data, plot=plt)
    plt.show()


def plot_scatter(x_data, y_data, col):
    plt.style.use("fivethirtyeight")

    plt.scatter(x_data, y_data, color='green', s=3.5)

    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.show()


def pre_process(x_data, y_data, val_ratio, for_model, is_duan):
    cols_to_select = []
    if for_model == "xg":
        cols_to_select = cols_to_select_xg
    elif for_model == "rf":
        cols_to_select = cols_to_select_rf

    print(cols_to_select)
    print(f"Cols to Select {cols_to_select.__len__()}")
    cat_variables = x_data.select_dtypes(include="object").columns

    print("1")
    print("**********************************")
    print(f"imputing not impute cat")
    temp = x_data[not_impute_cat]
    x_data[not_impute_cat] = fill_missing(temp, "zero")
    print("**********************************")

    print("2")
    print("**********************************")
    print(f"impute numeric vars")
    num_cols = x_data.select_dtypes(include="number").columns
    temp = x_data[num_cols]
    x_data[num_cols] = fill_missing(temp, "mean")
    print("**********************************")

    print("3")
    print("**********************************")
    print(f"impute categorical variables")
    cat_to_impute = [x for x in x_data.select_dtypes(include="object").columns if x not in not_impute_cat]
    temp = x_data[cat_to_impute]
    x_data[cat_to_impute] = fill_missing(temp, "frequent")
    print("**********************************")

    print(f"After imputing {x_data.isna().sum()}")

    print("A")
    print("**********************************")
    print(f"Remove Outliers SalePrice")
    x_data, y_data = remove_sale_price_outliers(x_data, y_data)
    print("**********************************")

    print("B")
    print("**********************************")
    print(f"Log Transform SalePrice")
    if not is_duan:
        y_data = log_transform_sale_price(y_data)
    print("**********************************")

    print("C")
    print("**********************************")
    print(f"Remove Outliers GrLivArea")
    x_data, y_data = remove_outliers(x_data, y_data, 2, 0, "GrLivArea")
    print("**********************************")

    print("D")
    print("**********************************")
    print(f"Log Transform GrLivArea")
    x_data["GrLivArea"] = log_transform_sale_price(x_data["GrLivArea"])
    print("**********************************")

    print("E")
    print("**********************************")
    print(f"Log Transform TotalBsmtSF")
    x_data = log_transform_zeroed_vals(x_data, "HasBsmt", "TotalBsmtSF")
    print("**********************************")

    print("F")
    print("**********************************")
    print(f"Log Transform Remaining")
    for col in skewed_cols:
        x_data[col] = boxcox1p(x_data[col], boxcox_normmax(x_data[col] + 1))
    print("**********************************")

    print("4")
    print("**********************************")
    print(x_data.shape)
    print(f"discretize categorical variables")

    def discretize_garage_type(value):
        if value == 0:
            return 0
        else:
            return 1

    x_data["GarageType"] = x_data["GarageType"].apply(discretize_garage_type)
    print("**********************************")

    print("5")
    print("Start Ordinal Encoding")
    x_data = encode_vars(x_data)
    print("Ordinal Encoding Done")
    print("**********************************")

    print("6")
    print("**********************************")
    print(x_data.shape)
    print("Manual Label Encoding")
    encoder = LabelEncoder()
    cols_to_encode = x_data.select_dtypes(include="object").columns
    temp = x_data[cols_to_encode]
    for col in cols_to_encode:
        if col == "Fence":
            continue
        x_data[col] = encoder.fit_transform(temp[col])
    print(x_data.shape)
    print("**********************************")
    print(f"All encoding done! categorical variables  {x_data.select_dtypes('object').shape}")
    print("Start Feature Selection!")
    print(cols_to_select)
    print(cols_to_select.__len__())
    cols_to_drop = [x for x in x_data.columns if x not in cols_to_select]
    print(f"cols to keep size {cols_to_select.__len__()}")
    print(f"cols to drop size {cols_to_drop.__len__()}")

    x_data.drop(labels=cols_to_drop, axis=1, inplace=True)
    print(x_data.shape)
    print("Feature selection done!")
    print("**********************************")

    print("7")
    print("**********************************")
    print(x_data.shape)
    print("Start one hot encoding ")
    print(f"number of features before one hot encoding {x_data.columns.shape[0]}")
    one_hot = [x for x in cat_variables if x not in ordinal_vars]
    one_hot = [x for x in one_hot if x in cols_to_select]
    idxs = []
    for x in one_hot:
        idxs.append(x_data.columns.get_loc(x))

    print(f"number of cols to one hot {idxs.__len__()}")

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(categories='auto'), idxs)],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    x_data = ct.fit_transform(x_data)
    print(f"number of features after one hot encoding {x_data.shape[1]}")
    print("End one hot encoding ")
    print("**********************************")

    print("8")
    print("**********************************")
    print("Scaling and Normalizing Features")
    scaler = StandardScaler(with_mean=False)
    print(x_data.shape)
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)
    print("**********************************")

    print("9")
    print("**********************************")
    print("Split Dataset")
    x_test = x_data[1448:, :]

    x_data = x_data[:1448, :]
    y_data = y_data[:1448]

    print(f"Train Ratio {1 - val_ratio}")
    print(f"Val Ratio {val_ratio}")

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=val_ratio)

    print(f"Train Size {x_train.shape[0]}")
    print(f"Val Ratio {x_val.shape[0]}")
    print("**********************************")
    print(f"Pre-processing Done for {for_model} ")
    return x_train, x_val, y_train, y_val, x_test
