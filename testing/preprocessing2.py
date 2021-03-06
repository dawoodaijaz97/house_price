import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from feature_selection import constant_filter, quasi_constant_filter, correlated_filter, forward_feature_selection, \
    tree_derived_feature_importance

not_impute_cat = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
                  "GarageType",
                  "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

ordinal_vars = ["Street", "Alley", "LotShape", "Utilities", "LandSlope", "ExterQual", "ExterCond",
                "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir",
                "KitchenQual", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", ]

discretize = ["GarageType"]


def fill_missing(x_data, strategy):
    if strategy == "mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    elif strategy == "zero":
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    elif strategy == "frequent":
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant")

    return imputer.fit_transform(x_data)


def pre_process(x_data, y_data, val_ratio):

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

    print("4")
    print("**********************************")

    print(f"discretize categorical variables")

    def discretize_garage_type(value):
        if value == 0:
            return 0
        else:
            return 1

    x_data["GarageType"] = x_data["GarageType"].apply(discretize_garage_type)
    print("**********************************")

    print("5")
    print("**********************************")

    print(f"encode ordinal Street")

    def ordinal_encode_street(val):
        if val == "Grvl":
            return 0
        else:
            return 1

    x_data["Street"] = x_data["Street"].apply(ordinal_encode_street)

    print("**********************************")

    print("6")
    print("**********************************")

    print(f"encode ordinal Alley")

    def ordinal_encode_alley(val):
        if val == "Grvl":
            return 1
        elif val == "Pave":
            return 2
        else:
            return 0

    x_data["Alley"] = x_data["Alley"].apply(ordinal_encode_alley)
    print("**********************************")

    print("7")
    print("**********************************")

    print(f"encode ordinal LotShape")

    def ordinal_encode_lotshape(val):
        if val == "Reg":
            return 4
        elif val == "IR1":
            return 3
        elif val == "IR2":
            return 2
        elif val == "IR3":
            return 1

    x_data["LotShape"] = x_data["LotShape"].apply(ordinal_encode_lotshape)

    print("**********************************")

    print("8")
    print("**********************************")

    print("encode ordinal Utilities")

    def ordinal_encode_utilities(val):

        if val == "ELO":
            return 1
        elif val == "NoSeWa":
            return 2
        elif val == "NoSewr":
            return 3
        elif val == "AllPub":
            return 4

    x_data["Utilities"] = x_data["Utilities"].apply(ordinal_encode_utilities)

    print("**********************************")

    print("9")
    print("**********************************")

    print("encode ordinal LandSlope")

    def ordinal_encode_landslope(val):

        if val == "Gtl":
            return 3
        elif val == "Mod":
            return 2
        elif val == "Sev":
            return 1

    x_data["LandSlope"] = x_data["LandSlope"].apply(ordinal_encode_landslope)
    print("**********************************")

    print("10")
    print("**********************************")

    print("encode ordinal ExterQual")

    def ordinal_encode_exterqual(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1

    x_data["ExterQual"] = x_data["ExterQual"].apply(ordinal_encode_exterqual)

    print("**********************************")

    print("11")
    print("**********************************")

    print("encode ordinal ExterCond")

    def ordinal_encode_extercond(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1

    x_data["ExterCond"] = x_data["ExterCond"].apply(ordinal_encode_extercond)
    print("**********************************")

    print("12")
    print("**********************************")

    print("ordinal encode BsmtQual")

    def ordinal_encode_bsmtqual(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1
        else:
            return 0

    x_data["BsmtQual"] = x_data["BsmtQual"].apply(ordinal_encode_bsmtqual)

    print("**********************************")

    print("13")
    print("**********************************")

    print("ordinal encdoe BsmtCond")

    def ordinal_encode_bsmtcond(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1
        else:
            return 0

    x_data["BsmtCond"] = x_data["BsmtCond"].apply(ordinal_encode_bsmtcond)

    print("**********************************")

    print("14")
    print("**********************************")

    print("ordinal encode BsmtExposure")

    def ordinal_encode_bsmtexposure(val):

        if val == "Gd":
            return 4
        elif val == "Av":
            return 3
        elif val == "Mn":
            return 2

        elif val == "No":
            return 1
        else:
            return 0

    x_data["BsmtExposure"] = x_data["BsmtExposure"].apply(ordinal_encode_bsmtexposure)

    print("**********************************")

    print("15")
    print("**********************************")

    print("ordinal encode BsmtFinType1")

    def ordinal_encode_bsmtfintype1(val):

        if val == "GLQ":
            return 6
        elif val == "ALQ":
            return 5
        elif val == "BLQ":
            return 4

        elif val == "Rec":
            return 3

        elif val == "LwQ":
            return 2

        elif val == "Unf":
            return 1
        else:
            return 0

    x_data["BsmtFinType1"] = x_data["BsmtFinType1"].apply(ordinal_encode_bsmtfintype1)

    print("**********************************")

    print("16")
    print("**********************************")

    print("ordinal encode BsmtFinType2")

    def ordinal_encode_bsmtfintype2(val):

        if val == "GLQ":
            return 6
        elif val == "ALQ":
            return 5
        elif val == "BLQ":
            return 4

        elif val == "Rec":
            return 3

        elif val == "LwQ":
            return 2

        elif val == "Unf":
            return 1
        else:
            return 0

    x_data["BsmtFinType2"] = x_data["BsmtFinType2"].apply(ordinal_encode_bsmtfintype2)

    print("17")
    print("**********************************")

    print("ordinal encode HeatingQC")

    def ordinal_encode_heatingqc(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4

        elif val == "TA":
            return 3
        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1

    x_data["HeatingQC"] = x_data["HeatingQC"].apply(ordinal_encode_heatingqc)

    print("**********************************")

    print("18")
    print("**********************************")
    print("ordinal encode KitchenQual")

    def ordinal_encode_kitchenqual(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4

        elif val == "TA":
            return 3
        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1
        else:
            return 0

    x_data["KitchenQual"] = x_data["KitchenQual"].apply(ordinal_encode_kitchenqual)
    print("**********************************")

    print("19")
    print("**********************************")
    print("ordinal encode FireplaceQu")

    def ordinal_encode_fireplacequ(val):

        if val == "Ex":
            return 5
        elif val == "Gd":
            return 4

        elif val == "TA":
            return 3
        elif val == "Fa":
            return 2

        elif val == "Po":
            return 1
        else:
            return 0

    x_data["FireplaceQu"] = x_data["FireplaceQu"].apply(ordinal_encode_fireplacequ)

    print("**********************************")

    print("20")
    print("**********************************")
    print("ordinal encode GarageFinish")

    def ordinal_encode_garagefinish(val):
        if val == "Fin":
            return 3

        elif val == "RFn":
            return 2
        elif val == "Unf":
            return 1
        else:
            return 0

    x_data["GarageFinish"] = x_data["GarageFinish"].apply(ordinal_encode_garagefinish)

    print("**********************************")

    print("21")
    print("**********************************")
    print("ordinal encode GarageQual")

    def ordinal_encode_garagequal(val):
        if val == "Ex":
            return 5

        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2
        elif val == "Po":
            return 1
        else:
            return 0

    x_data["GarageQual"] = x_data["GarageQual"].apply(ordinal_encode_garagequal)

    print("**********************************")

    print("22")
    print("**********************************")
    print("ordinal encode GarageCond")

    def ordinal_encode_garagecond(val):
        if val == "Ex":
            return 5

        elif val == "Gd":
            return 4
        elif val == "TA":
            return 3

        elif val == "Fa":
            return 2
        elif val == "Po":
            return 1
        else:
            return 0

    x_data["GarageCond"] = x_data["GarageCond"].apply(ordinal_encode_garagecond)

    print("**********************************")

    print("23")
    print("**********************************")
    print("ordinal encode PoolQC")

    def ordinal_encode_poolqc(val):
        if val == "Ex":
            return 5

        elif val == "Gd":
            return 3
        elif val == "TA":
            return 2

        elif val == "Fa":
            return 1
        else:
            return 0

    x_data["PoolQC"] = x_data["PoolQC"].apply(ordinal_encode_poolqc)

    print("**********************************")

    print("24")
    print("**********************************")
    print("ordinal encode CentralAir")

    def ordinal_encode_centralair(val):
        if val == "Y":
            return 1

        elif val == "N":
            return 0

    x_data["CentralAir"] = x_data["CentralAir"].apply(ordinal_encode_centralair)

    print("**********************************")

    print("25")

    print("**********************************")
    print("label encode Fence")

    def label_encode_fence(val):
        if val == "GdPrv":
            return 4
        elif val == "MnPrv":
            return 3
        elif val == "GdWo":
            return 2
        elif val == "MnWw":
            return 1

        else:
            return 0

    x_data["Fence"] = x_data["Fence"].apply(label_encode_fence)

    print("**********************************")

    print("26")

    print("**********************************")
    print("label encode MiscFeature")

    def label_encode_miscfeature(val):
        if val == "Elev":
            return 5
        elif val == "Gar2":
            return 4
        elif val == "Othr":
            return 3
        elif val == "Shed":
            return 2

        elif val == "Tenc":
            return 1

        else:
            return 0

    x_data["MiscFeature"] = x_data["MiscFeature"].apply(label_encode_miscfeature)

    print("**********************************")

    print(f"Number of categorical variables after ordinal encoding {x_data.select_dtypes(include='object').shape}")

    print("27")
    print("**********************************")
    print("Manual Label Encoding")
    encoder = LabelEncoder()
    cols_to_encode = x_data.select_dtypes(include="object").columns
    temp = x_data[cols_to_encode]
    for col in cols_to_encode:
        if col == "Fence":
            continue
        x_data[col] = encoder.fit_transform(temp[col])
    print("**********************************")
    print(f"All encoding done! categorical variables  {x_data.select_dtypes('object').shape}")
    print("Start Feature Selection!")
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=val_ratio)
    print("28")
    print("**********************************")

    print("Constant features filter ")
    x_train = constant_filter(x_train)
    print(x_train.shape)
    print("**********************************")

    print("29")
    print("**********************************")

    print("Quasi Constant feature filter")
    x_train = quasi_constant_filter(x_train, threshold=0.01)
    print(x_train.shape)
    print("**********************************")

    print("30")
    print("**********************************")

    print("Correlated feature filter")

    x_train = correlated_filter(x_train,correlation_threshold=0.8)
    print(x_train.shape)
    print("**********************************")

    print("31")
    print("**********************************")

    print("Tree based feature importance")
    x_train = tree_derived_feature_importance(x_train, y_train, 55)
    print(x_train.shape)
    print("**********************************")

    print("32")
    print("**********************************")

    print("Forward Feature Selection")
    x_train = forward_feature_selection(x_train,y_train,45)
    print(x_train.shape)
    print("**********************************")

    print(f"Cols to select after feature selection {x_train.columns}")



def main():
    path = "../data/train.csv"

    raw_data = pd.read_csv(path)

    x_data = raw_data.iloc[:,1:-1]
    y_data = raw_data.iloc[:,-1]

    pre_process(x_data,y_data,0.1)




if __name__ == "__main__":
    main()