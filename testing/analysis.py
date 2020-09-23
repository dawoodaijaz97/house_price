import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tabulate as tb

raw_data = pd.read_csv("./data/train.csv")

print(raw_data.columns)
print(raw_data.columns.shape)


def check_col_names(name,data):
    print(f"{name}: {data[name]}")


categorical_features = ["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","Utilities",
                        "LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType",
                        "HouseStyle","OverallQual","OverallCond","RoofStyle","RoofMatl",
                        "Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation",
                        "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating",
                        "HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu",
                        "GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence",
                        "MiscFeature","SaleType","SaleCondition"
                        ]
numerical_features = ["LotFrontage","LotArea","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","BsmtFinSF2",
                      "BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath",
                      "BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",
                      "GarageYrBlt","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch",
                      "ScreenPorch","PoolArea","MiscVal","MoSold","YrSold","SalePrice"]


ordinal_vars = ["Street","Alley","LotShape","LandContour","Utilities","LandSlope","ExterQual","ExterCond",
                "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","CentralAir",
                "KitchenQual","FireplaceQu","GarageFinish","GarageQual","GarageCond","PoolQC",""]
discretize = ["GarageType"]

print(categorical_features.__len__())
print(numerical_features.__len__())






def create_bar_graph(data_series,name):
    plt.figure(figsize=(9, 3))
    counts = data_series.value_counts()

    plt.bar(counts.index,counts.values)
    plt.xticks(counts.index,counts.index.values)

    plt.ylabel("Frequency")
    plt.xlabel(name)
    plt.show()


def create_histogram(data_series,name):

    plt.figure(figsize=(9, 8))
    plt.hist(data_series)
    plt.ylabel("Frequency")
    plt.xlabel(name)
    plt.show()


def scatter_plot(data_series,name):
    plt.figure(figsize=(9,8))
    plt.scatter(data_series[name],data_series["SalePrice"],s=5,cmap="summer")
    plt.ylabel("SalePrice")
    plt.xlabel(name)
    plt.show()

def plot_r(features):
    r_vals = []
    for feature in features:
        correlation_coefficient = np.corrcoef(raw_data[feature], raw_data["SalePrice"])
        r_vals.append(correlation_coefficient[0][1])

    print(r_vals.__len__())

    plt.figure(figsize=(9,8))
    plt.bar(features,r_vals)
    plt.xticks(features,features)
    plt.show()

def visualize():

    # BAR GRAPH FOR CATEGORICAL FEATURES
    # for feature in categorical_features:
    #     column = raw_data[feature]
    #     create_bar_graph(column,feature)

    # HISTOGRAM GRAPH FOR NUMERICAL FEATURES
    # for feature in numerical_features:
    #     column = raw_data[feature]
    #     create_histogram(column,feature)

    # SCATTER PLOT FOR NUMERICAL  FEATURES
    # for feature in numerical_features:
    #     data_series = raw_data[[feature,"SalePrice"]]
    #     scatter_plot(data_series,feature)

    # R VALS PLOT FOR NUMERICAL FEATURES
    plot_r(numerical_features)


def main():
    # visualize()
    column  = raw_data["SaleType"]

    values = column.value_counts()

    print((values/raw_data.shape[0]).sort_values(ascending=False).values[0])

if __name__ == "__main__":
    main()