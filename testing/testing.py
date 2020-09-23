import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile
import numpy as np
import tabulate as tb
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def fill_missing(x_data, strategy):
    if strategy == "mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    elif strategy == "zero":
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    elif strategy == "frequent":
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant")

    return imputer.fit_transform(x_data)


raw_data = pd.read_csv("./data/train.csv")
raw_data2 = pd.read_csv("./data/test.csv")

x_data = raw_data.iloc[:,1:-1]
y_data = raw_data["SalePrice"]

x_test = raw_data2.iloc[:,1:]

one_hot = ['MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'Exterior2nd',
           'MasVnrType',
           'Foundation', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']


drop_cols = [x for x in x_data.columns if x not in one_hot]

x_data.drop(labels=drop_cols,axis=1,inplace=True)
x_test.drop(labels=drop_cols,axis=1,inplace=True)



missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
temp = x_data
x_data = missingvalues.fit_transform(temp)
encoder = LabelEncoder()

for x in range(x_data.shape[1]):
    x_data[:,x] = encoder.fit_transform(x_data[:,x])

tot_ohe = 0
for x in range(x_data.shape[1]):
    tot_ohe = tot_ohe + pd.get_dummies(x_data[:, x]).shape[1]

print(tot_ohe)

missingvalues2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
temp2 = x_test
x_test = missingvalues.fit_transform(temp2)
encoder2 = LabelEncoder()


for x in range(x_test.shape[1]):
    x_test[:,x] = encoder2.fit_transform(x_test[:,x])

tot_ohe = 0
for x in range(x_test.shape[1]):
    tot_ohe = tot_ohe + pd.get_dummies(x_test[:, x]).shape[1]

print(tot_ohe)
