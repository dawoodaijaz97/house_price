import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_hist(data,bins):
    desc = data.describe()
    mean = desc["mean"]
    median = desc["50%"]
    val = data.values
    plt.style.use("fivethirtyeight")

    plt.hist(val,bins=bins,edgecolor='black')
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.axvline(median,color='red',label="Median Sale Price",linewidth=2)
    plt.axvline(mean,color='yellow',label="Mean Sale Price",linewidth=2)

    plt.show()

def main():
    path1 = "./data/train.csv"
    raw_data = pd.read_csv(path1)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    print(x_data.shape)
    print(y_data.shape)
    print()

    print(y_data.describe())
    plot_hist(y_data,20)





if __name__ == "__main__":
    main()
