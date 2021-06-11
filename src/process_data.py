"""
Data Cleaning Steps:
1. Data reduction 
- features with large single values (cardinality = 0)
- highly correlated features
- features with large missing values
- removing outliers

2. Data cleaning
- fill missing valuies
- encoding categorical features
"""

# Importing the libraries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

# logging.basicConfig(level=logging.DEBUG, filename='app.log',
#                     filemode='w', format='%(asctime)s - %(message)s')
logging.basicConfig(level=logging.DEBUG)


def read_data(csv_file: str) -> pd.DataFrame:
    """
    Validate and check that necessary columns are present
    Columns required are:
     - xxx, yy, zzz
    Check for encoding UTF-8, otherwise attempt to convert
    """
    return pd.read_csv(csv_file, index_col='Id')


def split_data(housing_df: pd.DataFrame, test_size: float) -> pd.DataFrame:
    X = housing_df.iloc[:, :-1]
    y = housing_df.iloc[:, -1]
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def standardize_data(df):
    df = df.copy()
    cont_features = ['BsmtTotalArea', 'LotTotalArea', 'OpenPorchSF',
                     'WoodDeckSF', 'GrLivArea', '2ndFlrSF', 'MasVnrArea']
    X_features = df[cont_features]
    sc_X = StandardScaler()
    X_features = sc_X.fit_transform(X_features.values)
    df[cont_features] = X_features
    return df


def drop_features(housing_df):
    housing_df.drop(['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                     'MiscVal', 'BsmtHalfBath', 'KitchenAbvGr', 'Street', 'Utilities', 'Condition2',
                     'RoofMatl', 'Heating', 'Functional', 'GarageQual', 'GarageCond', 'GarageYrBlt', '1stFlrSF',
                     'TotRmsAbvGrd', 'GarageArea', 'PoolQC', 'MiscFeature', 'Alley',
                     'Fence', 'FireplaceQu'], axis=1, inplace=True)
    return housing_df


def create_features(housing_df):
   # feature engineering
    housing_df['LotTotalArea'] = housing_df['LotFrontage'] + \
        housing_df['LotArea']
    housing_df.drop(['LotFrontage', 'LotArea'], axis=1, inplace=True)

    housing_df['BsmtTotalArea'] = housing_df['TotalBsmtSF'] + \
        housing_df['BsmtUnfSF'] + housing_df['BsmtFinSF1']
    housing_df.drop(['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1'],
                    axis=1, inplace=True)

    housing_df['TotalBathAbvGr'] = housing_df['FullBath'] + \
        housing_df['HalfBath']
    housing_df.drop(['FullBath', 'HalfBath'], axis=1, inplace=True)

    return housing_df


def handle_missing_data(housing_df):
    cols_with_missing = [
        col for col in housing_df.columns if housing_df[col].isnull().any()]

    for feature in cols_with_missing:
        if housing_df[feature].dtype != "object":
            housing_df[feature].fillna(
                housing_df[feature].mode()[0], inplace=True)
        elif housing_df[feature].dtype == "object":
            housing_df[feature].fillna("NA", inplace=True)

    return housing_df


def get_continuous_feature(housing_df):
    numerical_cols = [
        col for col in housing_df.columns if housing_df[col].dtype in ['int64', 'float64']]
    numerical_cols_discrete = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'FullBath', 'HalfBath',
                               'BedroomAbvGr', 'Fireplaces', 'GarageCars', 'MoSold', 'YearBuilt', 'YrSold', 'YearRemodAdd', 'MSSubClass']
    numerical_cols_continuous = []
    for i in numerical_cols:
        if i not in numerical_cols_discrete:
            numerical_cols_continuous.append(i)
    return numerical_cols_continuous


def get_dummy_features(housing_df):
    nominal_categorical_features = [
        col for col in housing_df.columns if housing_df[col].dtype in ['object']]

    # ohe = OneHotEncoder(handle_unknown='ignore')
    # ohe.fit(housing_df[nominal_categorical_features])
    # transformed = ohe.transform(housing_df[nominal_categorical_features])
    # ohe_df = pd.DataFrame.sparse.from_spmatrix(transformed, columns=ohe.get_feature_names())

    df_dummy = pd.get_dummies(
        housing_df[nominal_categorical_features], prefix_sep="__", columns=nominal_categorical_features)

    cat_dummies = [col for col in df_dummy if "__" in col and col.split(
        "__")[0] in nominal_categorical_features]

    cat_dummies_file = open("./model/processed_columns.txt", "w")
    for element in cat_dummies:
        cat_dummies_file.write(element + "\n")
    cat_dummies_file.close()

    processed_columns = list(df_dummy.columns[:])

    processed_columns_file = open("./model/processed_columns.txt", "w")
    for element in processed_columns:
        processed_columns_file.write(element + "\n")
    processed_columns_file.close()

    housing_df.drop(
        housing_df[nominal_categorical_features], axis=1, inplace=True)
    housing_df = pd.concat([df_dummy, housing_df], axis=1)

    return housing_df


def get_dummy_features_for_test(housing_df):
    cat_dummies = []
    # open file and read the content in a list
    with open('./model/processed_columns.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            feat = line[:-1]
            # add item to the list
            cat_dummies.append(feat)

    processed_columns = []
    # open file and read the content in a list
    with open('./model/processed_columns.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            feat = line[:-1]
            # add item to the list
            processed_columns.append(feat)

    nominal_categorical_features = [
        col for col in housing_df.columns if housing_df[col].dtype in ['object']]
    df_test_dummy = pd.get_dummies(
        housing_df, prefix_sep="__", columns=nominal_categorical_features)

    # Remove additional columns
    for col in df_test_dummy.columns:
        if ("__" in col) and (col.split("__")[0] in nominal_categorical_features) and col not in cat_dummies:
            #print("Removing additional feature {}".format(col))
            df_test_dummy.drop(col, axis=1, inplace=True)

    for col in cat_dummies:
        if col not in df_test_dummy.columns:
            #print("Adding missing feature {}".format(col))
            df_test_dummy[col] = 0

    df_test_dummy = df_test_dummy[processed_columns]

    housing_df.drop(
        housing_df[nominal_categorical_features], axis=1, inplace=True)
    housing_df = pd.concat([df_test_dummy, housing_df], axis=1)

    return housing_df


def ordinal_encode_features(housing_df):
    ordinal_qual_cond_cols = ['ExterQual', 'ExterCond',
                              'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual']
    qual_cond_sort = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    encoder1 = OrdinalEncoder(categories=[
                              qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort])
    encode_qual_cond_df = encoder1.fit_transform(
        housing_df[ordinal_qual_cond_cols])  # ugly array
    encode_qual_cond_df = pd.DataFrame(
        encode_qual_cond_df, columns=ordinal_qual_cond_cols)  # convert to df
    housing_df.reset_index(inplace=True)
    housing_df.drop(housing_df[ordinal_qual_cond_cols], axis=1, inplace=True)
    housing_df = pd.concat([encode_qual_cond_df, housing_df], axis=1)

    ordinal_bsmt_fin_cols = ['BsmtFinType1', 'BsmtFinType2']
    qual_bsmt_fin_sort = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    encoder2 = OrdinalEncoder(
        categories=[qual_bsmt_fin_sort, qual_bsmt_fin_sort])

    encode_bsmt_fin_df = encoder2.fit_transform(
        housing_df[ordinal_bsmt_fin_cols])
    encode_bsmt_fin_df = pd.DataFrame(
        encode_bsmt_fin_df, columns=ordinal_bsmt_fin_cols)
    housing_df.drop(housing_df[ordinal_bsmt_fin_cols], axis=1, inplace=True)
    housing_df = pd.concat([encode_bsmt_fin_df, housing_df], axis=1)

    bsmt_exposure_sort = ['NA', 'No', 'Mn', 'Av', 'Gd']
    encoder3 = OrdinalEncoder(categories=[bsmt_exposure_sort])
    encode_bsmt_exposure_df = encoder3.fit_transform(
        housing_df[['BsmtExposure']])
    encode_bsmt_exposure_df = pd.DataFrame(
        encode_bsmt_exposure_df, columns=['BsmtExposure_E'])

    housing_df.drop(['BsmtExposure'], axis=1, inplace=True)
    housing_df = pd.concat([encode_bsmt_exposure_df, housing_df], axis=1)

    return housing_df


def clean_data(housing_df: pd.DataFrame, data) -> pd.DataFrame:
    housing_df = drop_features(housing_df)
    housing_df = create_features(housing_df)
    housing_df = handle_missing_data(housing_df)
    housing_df = ordinal_encode_features(housing_df)
    if data == "train":
        housing_df = get_dummy_features(housing_df)
    elif data == "predict":
        housing_df = get_dummy_features_for_test(housing_df)
    housing_df.drop('Id', axis=1, inplace=True)
    if data == "train":
        # moving saleprice to last column of df
        salePrice = housing_df.pop('SalePrice')
        housing_df['saleprice'] = salePrice
    return housing_df


def get_bin(housing_df_c: pd.DataFrame) -> pd.DataFrame:
    bin_labels = [0, 1, 3, 4]
    price_25_per = np.percentile(housing_df_c['saleprice'], 25)
    price_50_per = np.percentile(housing_df_c['saleprice'], 50)
    price_75_per = np.percentile(housing_df_c['saleprice'], 75)
    price_100_per = np.percentile(housing_df_c['saleprice'], 100)

    cut_bins = [0, price_25_per, price_50_per, price_75_per, price_100_per]
    housing_df_c['saleprice_c'] = pd.cut(
        housing_df_c['saleprice'], bins=cut_bins, labels=bin_labels)

    housing_df_c.drop('saleprice', axis=1, inplace=True)

    return housing_df_c

    # def clean_data(housing_df: pd.DataFrame) -> pd.DataFrame:
    #     # a. large single values in each feature
    #     housing_df.drop(['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    #                      'MiscVal', 'BsmtHalfBath', 'KitchenAbvGr', 'Street', 'Utilities', 'Condition2',
    #                      'RoofMatl', 'Heating', 'Functional', 'GarageQual', 'GarageCond', 'GarageYrBlt', '1stFlrSF',
    #                      'TotRmsAbvGrd', 'GarageArea', 'PoolQC', 'MiscFeature', 'Alley',
    #                      'Fence', 'FireplaceQu'], axis=1, inplace=True)

    #     # b. dropping outliers > 3 std
    #     def remove_outlier(feat):
    #         outlier_threshold = [0.003, 0.997]
    #         min_threshold, max_threshold = feat.quantile(outlier_threshold)
    #         max_index = housing_df[feat >= max_threshold].index
    #         min_index = housing_df[feat < min_threshold].index
    #         housing_df.drop(min_index, inplace=True)
    #         housing_df.drop(max_index, inplace=True)
    #         return

    #     for feat in ['MSSubClass', 'BsmtFinSF1', 'TotalBsmtSF', 'LotArea']:
    #         housing_df[[feat]].apply(remove_outlier)

    #     housing_df['LotFrontage'].fillna(
    #         housing_df['LotFrontage'].mean(), inplace=True)
    #     housing_df[['LotFrontage']].apply(remove_outlier)

    #     # c. Filling missing values
    #     housing_df['MasVnrArea'] = housing_df['MasVnrArea'].fillna(0)
    #     housing_df[['MasVnrType', 'GarageFinish', 'GarageType', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual']] = housing_df[[
    #         'MasVnrType', 'GarageFinish', 'GarageType', 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual']].fillna('NA')
    #     housing_df['Electrical'] = housing_df['Electrical'].fillna(
    #         housing_df['Electrical'].mode()[0])

    #     # 4. Encoding features
    #     nominal_object_cols = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    #                            'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
    #                            'MasVnrType', 'Foundation', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
    #                            'PavedDrive', 'SaleType', 'SaleCondition']

    #     nominal_object_cols_encoded = pd.get_dummies(
    #         data=housing_df[nominal_object_cols], drop_first=True)

    #     housing_df.drop(['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',
    #                      'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
    #                      'MasVnrType', 'Foundation', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
    #                      'PavedDrive', 'SaleType', 'SaleCondition'], axis=1, inplace=True)

    #     housing_df = pd.concat([nominal_object_cols_encoded, housing_df], axis=1)

    #     ordinal_object_cols = ['ExterQual', 'ExterCond',
    #                            'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual']

    #     from sklearn.preprocessing import OrdinalEncoder
    #     qual_cond_sort = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    #     encoder1 = OrdinalEncoder(categories=[
    #                               qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort])
    #     encode_qual_cond_df = encoder1.fit_transform(
    #         housing_df[ordinal_object_cols])

    #     encode_qual_cond_df = pd.DataFrame(encode_qual_cond_df, columns=[
    #                                        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual'])
    #     housing_df.reset_index(inplace=True)
    #     housing_df.drop(['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    #                      'HeatingQC', 'KitchenQual'], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_qual_cond_df, housing_df], axis=1)

    #     qual_cond_bsmt_sort = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    #     encoder2 = OrdinalEncoder(
    #         categories=[qual_cond_bsmt_sort, qual_cond_bsmt_sort])
    #     encoder2.fit(housing_df[['BsmtFinType1', 'BsmtFinType2']])

    #     encode_bsmt_fin_df = pd.DataFrame(encoder2.transform(housing_df[[
    #                                       'BsmtFinType1', 'BsmtFinType2']]), columns=['BsmtFinType1_E', 'BsmtFinType2_E'])
    #     housing_df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_bsmt_fin_df, housing_df], axis=1)

    #     bsmt_exposure_sort = ['NA', 'No', 'Mn', 'Av', 'Gd']
    #     encoder3 = OrdinalEncoder(categories=[bsmt_exposure_sort])
    #     encoder3.fit(housing_df[['BsmtExposure']])

    #     encode_bsmt_exposure_df = pd.DataFrame(encoder3.transform(
    #         housing_df[['BsmtExposure']]), columns=['BsmtExposure_E'])
    #     housing_df.drop(['BsmtExposure'], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_bsmt_exposure_df, housing_df], axis=1)

    #     housing_df.drop('Id', axis=1, inplace=True)

    #     # print(housing_df.info())
    #     #logging.info('Clean DataFrame Shape: ', housing_df.shape)
    #     # logging.info('Clean train csv: <clean_train.csv>')
    #     logging.info('Data Cleaning: Completed.')

    #     return housing_df

    # def clean_data_new(housing_df: pd.DataFrame, data) -> pd.DataFrame:
    #     housing_df.drop(['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    #                      'MiscVal', 'BsmtHalfBath', 'KitchenAbvGr', 'Street', 'Utilities', 'Condition2',
    #                      'RoofMatl', 'Heating', 'Functional', 'GarageQual', 'GarageCond', 'GarageYrBlt', '1stFlrSF',
    #                      'TotRmsAbvGrd', 'GarageArea', 'PoolQC', 'MiscFeature', 'Alley',
    #                      'Fence', 'FireplaceQu'], axis=1, inplace=True)

    #     numerical_cols = [
    #         col for col in housing_df.columns if housing_df[col].dtype in ['int64', 'float64']]

    #     object_cols = [
    #         col for col in housing_df.columns if housing_df[col].dtype == "object"]

    #     numerical_cols_discrete = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'FullBath', 'HalfBath',
    #                                'BedroomAbvGr', 'Fireplaces', 'GarageCars', 'MoSold', 'YearBuilt', 'YrSold',
    #                                'YearRemodAdd', 'MSSubClass']

    #     numerical_cols_continuous = []
    #     for i in numerical_cols:
    #         if i not in numerical_cols_discrete:
    #             numerical_cols_continuous.append(i)

    #     # feature engineering
    #     housing_df['LotTotalArea'] = housing_df['LotFrontage'] + \
    #         housing_df['LotArea']
    #     housing_df.drop(['LotFrontage', 'LotArea'], axis=1, inplace=True)

    #     housing_df['BsmtTotalArea'] = housing_df['TotalBsmtSF'] + \
    #         housing_df['BsmtUnfSF'] + housing_df['BsmtFinSF1']
    #     housing_df.drop(['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1'],
    #                     axis=1, inplace=True)

    #     housing_df['TotalBathAbvGr'] = housing_df['FullBath'] + \
    #         housing_df['HalfBath']
    #     housing_df.drop(['FullBath', 'HalfBath'], axis=1, inplace=True)

    #     num_cols_with_missing = [
    #         col for col in housing_df.columns if housing_df[col].isnull().any()]

    #     for feature in num_cols_with_missing:
    #         if housing_df[feature].dtype != "object":
    #             housing_df[feature].fillna(
    #                 housing_df[feature].mode()[0], inplace=True)
    #         elif housing_df[feature].dtype == "object":
    #             housing_df[feature].fillna("NA", inplace=True)

    #     ordinal_qual_cond_cols = ['ExterQual', 'ExterCond',
    #                               'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual']
    #     qual_cond_sort = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    #     encoder1 = OrdinalEncoder(categories=[
    #                               qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort, qual_cond_sort])
    #     encode_qual_cond_df = encoder1.fit_transform(
    #         housing_df[ordinal_qual_cond_cols])  # ugly array
    #     encode_qual_cond_df = pd.DataFrame(
    #         encode_qual_cond_df, columns=ordinal_qual_cond_cols)  # convert to df
    #     housing_df.reset_index(inplace=True)
    #     housing_df.drop(housing_df[ordinal_qual_cond_cols], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_qual_cond_df, housing_df], axis=1)

    #     ordinal_bsmt_fin_cols = ['BsmtFinType1', 'BsmtFinType2']
    #     qual_bsmt_fin_sort = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    #     encoder2 = OrdinalEncoder(
    #         categories=[qual_bsmt_fin_sort, qual_bsmt_fin_sort])

    #     encode_bsmt_fin_df = encoder2.fit_transform(
    #         housing_df[ordinal_bsmt_fin_cols])
    #     encode_bsmt_fin_df = pd.DataFrame(
    #         encode_bsmt_fin_df, columns=ordinal_bsmt_fin_cols)
    #     housing_df.drop(housing_df[ordinal_bsmt_fin_cols], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_bsmt_fin_df, housing_df], axis=1)

    #     bsmt_exposure_sort = ['NA', 'No', 'Mn', 'Av', 'Gd']
    #     encoder3 = OrdinalEncoder(categories=[bsmt_exposure_sort])
    #     encode_bsmt_exposure_df = encoder3.fit_transform(
    #         housing_df[['BsmtExposure']])
    #     encode_bsmt_exposure_df = pd.DataFrame(
    #         encode_bsmt_exposure_df, columns=['BsmtExposure_E'])

    #     housing_df.drop(['BsmtExposure'], axis=1, inplace=True)
    #     housing_df = pd.concat([encode_bsmt_exposure_df, housing_df], axis=1)

    #     nominal_categorical_features = [
    #         col for col in housing_df.columns if housing_df[col].dtype in ['object']]

    #     nominal_categorical_features_dummy = pd.get_dummies(
    #         data=housing_df[nominal_categorical_features], drop_first=True, dtype='int64')

    #     housing_df.drop(
    #         housing_df[nominal_categorical_features], axis=1, inplace=True)
    #     housing_df = pd.concat(
    #         [nominal_categorical_features_dummy, housing_df], axis=1)

    #     housing_df.drop('Id', axis=1, inplace=True)

    #     if data == "train":
    #         salePrice = housing_df.pop('SalePrice')
    #         housing_df['saleprice'] = salePrice

    #     logging.info('Data Cleaning: Completed.')

    #     return housing_df

    # def standardize_data(df):
    #     # X_train = X_train.copy()
    #     # col_names = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
    #     #              'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']

    #     # X_train_features = X_train[col_names]
    #     # sc_X = StandardScaler()
    #     # X_train_features = sc_X.fit_transform(X_train_features.values)
    #     # X_train[col_names] = X_train_features

    #     # X_test = X_test.copy()
    #     # X_test_features = X_test[col_names]
    #     # X_test_features = sc_X.transform(X_test_features.values)
    #     # X_test[col_names] = X_test_features
    #     # return X_train, X_test

    #     # X = X.copy()
    #     # col_names = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
    #     #              'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']
    #     # X_features = X[col_names]
    #     # sc_X = StandardScaler()
    #     # X_features = sc_X.fit_transform(X_features.values)
    #     # X[col_names] = X_features

    #     df = df.copy()
    #     cont_features = ['BsmtTotalArea', 'LotTotalArea', 'OpenPorchSF',
    #                      'WoodDeckSF', 'GrLivArea', '2ndFlrSF', 'MasVnrArea']
    #     X_features = df[cont_features]
    #     sc_X = StandardScaler()
    #     X_features = sc_X.fit_transform(X_features.values)
    #     df[cont_features] = X_features

    #     return df
