import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tabulate
import scipy.stats as ss

from preprocessing2 import pre_process

not_impute_cat = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
                  "GarageType",
                  "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

ordinal_vars = ["Street", "Alley", "LotShape", "Utilities", "LandSlope", "ExterQual", "ExterCond",
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir",
                "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", ]
discretize = ["GarageType"]


def main():
    raw_data = pd.read_csv("./data/train.csv")
    print(f"Total number of variables: {raw_data.shape[1]}")
    print(f"Number of numerical variables: {raw_data.select_dtypes(include='number').columns.shape[0]}")
    print(f"Number of categorical variables: {raw_data.select_dtypes(include='object').columns.shape[0]}")
    print(f"Number of categorical variables not to compute: {not_impute_cat.__len__()}")
    print(f"Number of ordinal variables: {ordinal_vars.__len__()}")

    print(raw_data.shape)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    pre_process(x_data, y_data, 0.1)


if __name__ == "__main__":
    main()
