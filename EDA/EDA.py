import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

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
               'TotalBsmtSF',
               'MSSubClass',
               '1stFlrSF',
               'GrLivArea', ]


def plot_hist(data, bins, col):
    desc = data.describe()
    mean = desc["mean"]
    median = desc["50%"]
    val = data.values
    plt.style.use("fivethirtyeight")

    plt.hist(val, bins=bins, edgecolor='black')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.axvline(median, color='red', label="Median Sale Price", linewidth=2)
    plt.axvline(mean, color='yellow', label="Mean Sale Price", linewidth=2)

    plt.show()


def plot_scatter(x_data, y_data, col):
    plt.style.use("fivethirtyeight")

    plt.scatter(x_data, y_data, color='green', s=3.5)

    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.show()


def plot_boxplot(x, y, x_data, y_data):
    data = pd.concat((x_data, y_data), axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x, y, data=data)
    plt.show()


def plot_heatmap(x_data):
    corr_matrix = x_data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, square=True)
    plt.show()


def y_corr(raw_data):
    corr_matrix = raw_data.corr()
    corr_matrix = corr_matrix.nlargest(10, "SalePrice")["SalePrice"]
    print(corr_matrix)

    plt.subplots(figsize=(12, 9))
    plt.bar(corr_matrix.index, corr_matrix.values, width=0.5)
    plt.xticks(corr_matrix.index, corr_matrix.index.values)
    plt.show()


def plot_pairplot(data):
    plt.subplots(figsize=(12, 9))
    sns.pairplot(data, diag_kind="kde", size=2.5, kind="reg")
    plt.show()


def plot_bar(x_data):
    plt.subplots(figsize=(12, 9))
    plt.bar(x_data.index, x_data.values, width=0.5)
    plt.xticks(x_data.index, x_data.index.values)
    plt.show()


def get_missing_val_stats(x_data):
    null_vals = x_data.isnull().sum().sort_values(ascending=False)
    percentage = (x_data.isnull().sum() / x_data.isnull().count()).sort_values(ascending=False)
    return percentage


def dist_prob_plot(data, col):
    plt.subplots(figsize=(12, 9))
    sns.distplot(data[col], kde=True)
    plt.show()
    stats.probplot(data[col], plot=plt)
    plt.show()


def dist_prob_plot2(data):
    plt.subplots(figsize=(12, 9))
    sns.distplot(data, kde=True)
    plt.show()
    stats.probplot(data, plot=plt)
    plt.show()


def log_transform_series(series):
    data = np.log(series)
    return pd.Series(data)

def main():
    path1 = "./data/train.csv"

    raw_data = pd.read_csv(path1)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    # numeric_cols = x_data.select_dtypes(include="number").columns
    # print(numeric_cols)
    # skewed_cols = []
    # skew_val = []
    # for col in numeric_cols:
    #     print(col)
    #     skew = x_data[col].skew()
    #     if skew > 0.5 or skew < -0.5:
    #         skewed_cols.append(col)
    #         skew_val.append(skew)
    #
    # cols = pd.Series(skewed_cols, name="Skewed Cols")
    # val = pd.Series(skew_val, name="skew vals")
    #
    # df = pd.concat((cols, val), axis=1)
    # print(df.sort_values(by=['skew vals'], axis=0, ascending=False))

    x_data = x_data.fillna(0)


    for col in skewed_cols:
        x_data[col] = boxcox1p(x_data[col],boxcox_normmax(x_data[col] + 1))

    print(x_data["TotalBsmtSF"].skew())
    print(x_data["TotalBsmtSF"].describe())

    skewed_cols_g = []
    skew_val = []
    for col in skewed_cols:
        skew = x_data[col].skew()
        skewed_cols_g.append(col)
        skew_val.append(skew)

    cols = pd.Series(skewed_cols_g, name="Skewed Cols")
    val = pd.Series(skew_val, name="skew vals")

    df = pd.concat((cols, val), axis=1)
    print(df.sort_values(by=['skew vals'], axis=0, ascending=False))



    # corr = raw_data.corr()
    #
    # print(corr["SalePrice"].sort_values(ascending=False))

    # scaler = StandardScaler()
    # scaled_bsmtsf = scaler.fit_transform(np.reshape(x_data["TotalBsmtSF"].values,(-1,1)))
    # scaled_data = pd.Series(np.reshape(scaled_bsmtsf, (-1,)))
    # print(scaled_data.describe())
    # print(scaled_data.skew())
    # print(scaled_data.kurt())
    # dist_prob_plot2(scaled_data)
    # x_data["HasBsmt"] = pd.Series(len(x_data["TotalBsmtSF"]), index=x_data.index)
    # x_data["HasBsmt"] = 0
    # x_data.loc[x_data["TotalBsmtSF"] > 0, "HasBsmt"] = 1
    # x_data.loc[x_data["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(x_data["TotalBsmtSF"])
    # plot_hist(x_data.loc[x_data["TotalBsmtSF"]>0,"TotalBsmtSF"],50,"TotalBsmtSF")
    # print(x_data.loc[x_data["TotalBsmtSF"]>0,"TotalBsmtSF"].describe())
    # print(x_data.loc[x_data["TotalBsmtSF"]>0,"TotalBsmtSF"].skew())
    # print(x_data.loc[x_data["TotalBsmtSF"]>0,"TotalBsmtSF"].kurt())
    # dist_prob_plot2(x_data.loc[x_data["TotalBsmtSF"]>0,"TotalBsmtSF"])

    # log_trans = np.log(x_data["GrLivArea"])
    # print(log_trans.describe())
    # print(log_trans.skew())
    # print(log_trans.kurt())
    # dist_prob_plot2(log_trans)

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(np.reshape(x_data["GrLivArea"].values,(-1,1)))
    #
    # scaled_data = pd.Series(np.reshape(scaled_data,(-1,)))
    # print(scaled_data.describe())
    # print(scaled_data.skew())
    # print(scaled_data.kurt())
    # plot_hist(scaled_data,50,'GrLivArea')
    # dist_prob_plot2(scaled_data)
    #
    # print(x_data["GrLivArea"].isnull().sum())
    #
    # plot_scatter(x_data["GrLivArea"], y_data,'GrLivArea')
    #
    # corr = raw_data.corr()
    # print(corr)
    # print(corr["SalePrice"].sort_values(ascending=False))

    # print("Before Processing")
    # print(y_data.describe())
    #
    # print(y_data.skew())
    # print(y_data.kurt())
    #
    # plot_hist(y_data,50)
    # dist_prob_plot2(y_data)
    #
    # print("After Standard Scaling")
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(np.reshape(y_data.values, (-1, 1)))
    # scaled_data = pd.Series(np.reshape(scaled_data,(-1,)))
    #
    # print(scaled_data.describe())
    # print(scaled_data.skew())
    # print(scaled_data.kurt())
    #
    # plot_hist(scaled_data,50)
    # dist_prob_plot2(scaled_data)
    #
    # print("After Log Transform")
    # log_trans = np.log(y_data)
    #
    # log_trans = pd.Series(log_trans)
    #
    # print(log_trans.describe())
    # print(log_trans.skew())
    # print(log_trans.kurt())
    #
    # plot_hist(log_trans, 50)
    # dist_prob_plot2(log_trans)
    #
    # print("After Standard Scaling and  Log  Transform")
    # scaled_and_log = np.log(scaled_data)
    #
    # scaled_and_log = pd.Series(scaled_and_log)
    #
    # print(scaled_and_log.describe())
    # print(scaled_and_log.skew())
    # print(scaled_and_log.kurt())
    #
    # plot_hist(scaled_and_log, 50)
    # dist_prob_plot2(scaled_and_log)

    # plot_hist(y_data,50)

    # plot_scatter(x_data["GrLivArea"],y_data,"GrLivArea")
    # plot_scatter(x_data["TotalBsmtSF"],y_data,"TotalBsmtSF")

    # plot_boxplot("OverallQual","SalePrice",x_data["OverallQual"],y_data)

    # plot_boxplot("YearBuilt","SalePrice",x_data["YearBuilt"],y_data)

    # plot_heatmap(x_data)

    # plot_y_heatmap(raw_data)

    # y_corr(raw_data)

    important_cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF",
                      "FullBath", "TotRmsAbvGrd", "YearBuilt"]

    important_cols2 = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF",
                       "FullBath", "TotRmsAbvGrd", "YearBuilt"]
    # plot_pairplot(raw_data[important_cols])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #
    #     percentage = get_missing_val_stats(x_data)
    #     #plot_bar(percentage)
    #
    #     scaler = StandardScaler()
    #
    #     scaled_y = scaler.fit_transform(np.reshape(raw_data["SalePrice"].values,(-1,1)))
    #
    #     sale_price = pd.Series(np.reshape(scaled_y,(-1,)))
    #
    #     print(f"After Standard Scaling {sale_price.describe()}")
    #
    #     dist_prob_plot2(sale_price)
    #
    #
    #
    #
    #     print("Min 10")
    #     print(sale_price.iloc[:10,])
    #     print("Max 9")
    #     print(sale_price.iloc[-9:,])

    # for col in important_cols2:
    #     plot_scatter(x_data[col],y_data,col)

    # dist_prob_plot(raw_data,"SalePrice")
    # raw_data["SalePrice"] = np.log(raw_data["SalePrice"])
    # dist_prob_plot(raw_data, "SalePrice")

    # dist_prob_plot(raw_data,"GrLivArea")
    # raw_data["GrLivArea"] = np.log(raw_data["GrLivArea"])
    # dist_prob_plot(raw_data, "GrLivArea")

    # dist_prob_plot(raw_data,"TotalBsmtSF")

    # x_data["HasBsmt"] = pd.Series(len(x_data["TotalBsmtSF"]),index=x_data.index)
    # x_data["HasBsmt"] = 0
    # x_data.loc[x_data["TotalBsmtSF"]>0,"HasBsmt"] = 1
    # x_data.loc[x_data["HasBsmt"] == 1,"TotalBsmtSF"] = np.log(x_data["TotalBsmtSF"])
    #
    # raw_data["TotalBsmtSF"] = np.log(raw_data["TotalBsmtSF"])
    # dist_prob_plot(raw_data.loc[x_data["TotalBsmtSF"]>0], "TotalBsmtSF")
    #


if __name__ == "__main__":
    main()
