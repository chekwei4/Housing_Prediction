import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import logging
from . import process_data
from sklearn.svm import SVR
from statistics import mean
import xgboost as xgb
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import joblib

# TODO
# 1. refactor standardize_data method -> done
# 2. gridsearch for kernels, read from config file. use JSON -> done
# 3. return statement for train() method -> done
# 4. work on model_rf, model_sv for cv (no need for simplesplit) -> done
# 5. create joblib for best model - default for predict.


def read_config_param():
    f = open('config.json',)
    data = json.load(f)
    return data


def train(df_cleaned, model, cv):
    df_cleaned_scaled = process_data.standardize_data(df_cleaned)
    X = df_cleaned_scaled.iloc[:, :-1]
    y = df_cleaned_scaled.iloc[:, -1]
    if model == "rf":
        # return model_rf(X, y, cv)
        return model_rf_with_rscv(X, y, cv)
    # elif model == "svr":
        # return model_svr(X, y, cv)
    elif model == "xgb":
        # return model_xgb(X, y, cv)
        return model_xgb_with_rscv(X, y, cv)


def predict(df_cleaned, joblib_file):
    df_cleaned_scaled = process_data.standardize_data(df_cleaned)
    joblib_xgb_model = joblib.load(joblib_file)
    logging.info("Running XGB Model...")
    prediction_df = joblib_xgb_model.predict(df_cleaned_scaled)
    prediction_df = pd.DataFrame(prediction_df, columns=[
                                 "SalePrice"])
    # prediction_df.index = range(1461, prediction_df.shape[0]+1)
    prediction_df.index += 1461
    prediction_df.to_csv("./prediction/prediction.csv",
                         index_label="Id")
    logging.info("Prediction saved...")
    return


def model_rf_with_rscv(X, y, cv):
    rf_regressor = RandomForestRegressor()
    data = read_config_param()
    random_grid = data['model']['rf']
    # rf_random = RandomizedSearchCV(estimator=rf_regressor, param_distributions=random_grid,
    #                                n_iter=10, cv=cv, verbose=10, random_state=42, n_jobs=-1)
    rf_model = RandomizedSearchCV(estimator=rf_regressor, param_distributions=random_grid,
                                  n_iter=10, cv=cv, verbose=0, random_state=42, n_jobs=-1,
                                  scoring='neg_mean_absolute_error')
    logging.info("Running RandomizedSearchCV...")
    rf_model.fit(X, y)
    logging.info("RF Reg: Completed.")
    return rf_model.best_score_


def model_xgb_with_rscv(X, y, cv):
    # data_dmatrix = xgb.DMatrix(data=X, label=y)
    xgb_reg = XGBRegressor()
    data = read_config_param()
    random_grid = data['model']['xgb']
    xgb_model = RandomizedSearchCV(xgb_reg, param_distributions=random_grid, n_iter=10,
                                   scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, verbose=0)
    logging.info("Running RandomizedSearchCV ...")
    xgb_model.fit(X, y)
    logging.info("XGB Reg: Completed.")
    joblib_file = "./model/joblib_xgb_model.pkl"
    joblib.dump(xgb_model.best_estimator_, joblib_file)

    return xgb_model.best_score_


# def model_rf(X, y, cv):
#     rf_regressor = RandomForestRegressor(n_estimators=100)
#     # result_r2 = cross_val_score(estimator=rf_regressor,
#     #                             X=X, y=y, cv=kfold, scoring='r2')
#     result_mae = cross_val_score(
#         estimator=rf_regressor, X=X, y=y, cv=cv, scoring='neg_mean_absolute_error')
#     logging.info("RF CV: Completed.")
#     return mean(result_mae)


# def model_xgb(X, y, cv):
#     data_dmatrix = xgb.DMatrix(data=X, label=y)
#     params = {"objective": "reg:squarederror", 'colsample_bytree': 0.8, 'subsample': 0.8, 'learning_rate': 0.1,
#               'max_depth': 5, 'min_child_weight': 3, 'alpha': 10}
#     cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=cv,
#                         num_boost_round=200, early_stopping_rounds=10, metrics="mae", as_pandas=True, seed=42)
#     result_mae = cv_results["test-mae-mean"].tail(1)
#     return mean(result_mae)

# def model_svr(X, y, cv):
#     sv_regressor = SVR(kernel='rbf')
#     kfold = KFold(n_splits=cv)
#     # result_r2 = cross_val_score(estimator=sv_regressor,
#     #                             X=X, y=y, cv=kfold, scoring='r2')
#     result_mae = cross_val_score(
#         estimator=sv_regressor, X=X, y=y, cv=kfold, scoring='neg_mean_absolute_error')
#     logging.info("SV CV: Completed.")
#     return mean(result_mae)
