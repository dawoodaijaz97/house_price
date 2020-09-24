import numpy as np
import pandas as pd
from matplotlib import pyplot as pp


def main():
    path1 = "./data/train.csv"
    raw_data = pd.read_csv(path1)

    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data.iloc[:, -1]

    print(x_data.shape)
    print(y_data.shape)
    print()

    print(y_data.describe())


if __name__ == "__main__":
    main()
