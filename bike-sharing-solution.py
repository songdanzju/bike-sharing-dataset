#!/usr/bin/python
#-*- coding: utf8 -*-
#@author: jianzhong.chen68@gmail.com
"""cjzpy_ml_bike.py
function: machine learning Bike Sharing dataset
ML modules:
- DecisionTreeRegressor
- ExtraTreesRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- SVR
- GridSearchCV
- metrics.mean_squared_error
data sets:
- https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
usage: 
- python3 cjzpy_ml_bike.py
references:
- https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_data(fn):
    """ simply read csv file with pandas read_csv
        Parameters
        ----------
        fn: csv file name
        
        Returns
        -------
        df: pandas dataframe
    """
    df = pd.read_csv(fn)
#    print('df.head(5)\n', df.head(5))
    return df

def check_df(df):
    """ check if there are missing entries in df
        Parameters
        ----------
        df: pandas dataframe
    """
#    print(df[df.isnull().any(axis=1)])
    print(df.isnull().any(axis=0))
    assert df[df.isnull().any(axis=1)].index.tolist()==[]

def feature_engineering(df):
    """ generate new informative features for df
        Parameters
        ----------
        df: pandas dataframe
        
        Returns
        -------
        df: feature engineered dataframe
    """
    def cat_hr(x):
        """function for categorizing hr (by building decision tree)"""
        if x['hr'] <= 6.5:
            return 0
        elif x['hr'] <= 8.5:
            return 1
        elif x['hr'] <= 15.5:
            return 2
        elif x['hr'] <= 19.5:
            return 3
        elif x['hr'] <= 21.5:
            return 4
        else:
            return 5
    
    def cat_temp(x):
        """simple function for categrizing temp (by building decision tree)"""
        if x['temp'] <= 0.27:
            return 0
        elif x['temp'] <= 0.35:
            return 1
        elif x['temp'] <= 0.69:
            return 2
        else:
            return 3
    
    def cat_hum(x):
        """simple function for categrizing hum (by building decision tree)"""
        if x['hum'] <= 0.435:
            return 0
        elif x['hum'] <= 0.625:
            return 1
        elif x['hum'] <= 0.855:
            return 2
        else:
            return 3
    
    # parse dteday feature to generate new features
    dt = pd.DatetimeIndex(df['dteday'])
    df.set_index(dt, inplace=True)
#    df['date'] = dt.date
    df['day'] = dt.day
    df['month'] = dt.month
    df['year'] = dt.year
#    df['hour'] = dt.hour
    df['dow'] = dt.dayofweek
    df['woy'] = dt.weekofyear
    
    # the peak_hr distribution (bike_figure_1.pdf) 
    df['peak_hr'] = df[['hr', 'workingday']].apply(
        lambda x: (0, 1)[(x['workingday']==1 and (7<=x['hr']<=9 or 17<=x['hr']<= 19)) or 
                         (x['workingday']==0 and 10<=x['hr']<=19)], axis = 1)
    
    # comfortable condition for riding (bike_figure_1.pdf)
    df['atemp_windspeed'] = df[['atemp', 'windspeed']].apply(
        lambda x: (0, 1)[(0.2537<=x['windspeed']<=0.2537 or 0.3284<=x['windspeed']<= 0.4925) and 
                         (0.6212<=x['atemp']<=0.6667)], axis = 1)
    
    # categorize 'hr', 'hum' and 'temp' with decision trees, done offline
    df['hr_cat'] = df[['hr']].apply(cat_hr, axis=1)
    df['temp_cat'] = df[['temp']].apply(cat_temp, axis=1)
    df['hum_cat'] = df[['hum']].apply(cat_hum, axis=1)
    
    return df
        
def split_data(df, features):
    """ split df[features] into train and test set with ShuffleSplit
        it also generates a new feature 'cnt_season' by grouping counts of four seasons
        Parameters
        ----------
        df: pandas dataframe
        features: a list of columns of df, the set of features in train set
        
        Returns
        -------
        df: dataframe + 'cnt_season'
        X_train, X_test, y_train, y_test: train set and test set for 'cnt' column
        y_train_cas, y_test_cas, y_train_reg, y_test_reg: train and test sets for 'casual' and 'registered' columns (not used in this study)
        time_test: datetime information of test set, for writing prediction results
    """
    ss = cross_validation.ShuffleSplit(len(df), n_iter=1, test_size=0.1, random_state=1234)
    for ind_train, ind_test in ss:
        # add a cnt_season column using groupby and join
        if 'cnt_season' not in df:
            season_gb = df.ix[ind_train, :].groupby('season')[['cnt']].agg(sum)
            season_gb.columns = ['cnt_season']
            df = df.join(season_gb, on='season')
        X_train = df.ix[ind_train, features].as_matrix()
        X_test = df.ix[ind_test, features].as_matrix()
        y_train = np.log1p(df.ix[ind_train, 'cnt'].as_matrix())
        y_test = np.log1p(df.ix[ind_test, 'cnt'].as_matrix())
        y_train_cas = np.log1p(df.ix[ind_train, 'casual'].as_matrix())
        y_train_reg = np.log1p(df.ix[ind_train, 'registered'].as_matrix())
        y_test_cas = np.log1p(df.ix[ind_test, 'casual'].as_matrix())
        y_test_reg = np.log1p(df.ix[ind_test, 'registered'].as_matrix())
        time_test = df.ix[ind_test, ['dteday', 'mnth', 'hr']].as_matrix()
    return df, X_train, X_test, y_train, y_test, y_train_cas, y_test_cas, y_train_reg, y_test_reg, time_test

def predict_evaluate(est, X_train, y_train, X_test, y_test):
    """ train/fit model on train set, predict test set, then calculate MSE
        Parameters
        ----------
        est: sklearn estimator / regressor
        X_train, y_train, X_test, y_test: train set and test set
        
        Returns
        -------
        y_pred: prediction of test set
        mse: MSE of y_pred vs. y_test
    """
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    return y_pred, mse

def test_run(fn, features, type):
    """ load dataset, build feature set, and do learning
        Parameters
        ----------
        fn: file name of dataset
        features: a list of list, each of which is a feature list for different models
        type: str for indicating feature set
        
        Returns
        -------
        predictions and feature-engineered dataset are saved to files
    """
    np.set_printoptions(precision=4)
    print('test_run ' + type)
    df = load_data(fn)
    check_df(df)
    df = feature_engineering(df)
    
    print(df.columns)
#    print(df.head())
#    print(df.groupby(['peak_hr'])['cnt'].agg(sum))
    y_pred_list = []
    for i, est in enumerate((
        DecisionTreeRegressor(min_samples_split=20),
        ExtraTreesRegressor(n_estimators=100, max_depth=None, min_samples_split=1, random_state=1234),
        RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=1234, min_samples_split=3, n_jobs=-1),
        GradientBoostingRegressor(n_estimators=150, max_depth=10, random_state=0, min_samples_leaf=20, learning_rate=0.1, subsample=0.7, loss='ls'),
        svm.SVR(C=30)
        )):
#        print(features[i])
        df, X_train, X_test, y_train, y_test, y_train_cas, y_test_cas, y_train_reg, y_test_reg, time_test = split_data(df, features=features[i])
        y_pred, mse = predict_evaluate(est, X_train, y_train, X_test, y_test)
        est_name = str(est).split('(')[0]
        print(type, est_name, np.round(mse, 4))
        """ feature importance
        if est_name != 'SVR':
            # print out feature importance
            sfi = sorted([(x[0], float('%.4f'%x[1])) for x in zip(features[i], est.feature_importances_)], key=lambda x: x[1], reverse=True)
            print(sfi)
            print([x[0] for x in sfi])
        """
        y_pred_list.append([est_name, mse, y_pred])

    # blending models
    y_pred_blend = np.log1p(.2*(np.exp(y_pred_list[2][2])-1) + .8*(np.exp(y_pred_list[3][2])-1))
    print(type+' blending: 0.2*'+y_pred_list[2][0]+' + 0.8*'+y_pred_list[3][0], metrics.mean_squared_error(y_test, y_pred_blend).round(4))
    y_pred_blend = np.log1p(.3*(np.exp(y_pred_list[1][2])-1) + .7*(np.exp(y_pred_list[3][2])-1))
    print(type+' blending: 0.3*'+y_pred_list[1][0]+' + 0.7*'+y_pred_list[3][0], metrics.mean_squared_error(y_test, y_pred_blend).round(4))
    y_pred_blend = np.log1p(.3*(np.exp(y_pred_list[3][2])-1) + .7*(np.exp(y_pred_list[4][2])-1))
    print(type+ ' blending: 0.2*'+y_pred_list[3][0]+' + 0.8*'+y_pred_list[4][0], metrics.mean_squared_error(y_test, y_pred_blend).round(4))
    y_pred_blend = np.log1p(.6*(np.exp(y_pred_list[3][2])-1) + .4*(np.exp(y_pred_list[4][2])-1))
    print(type+ ' blending: 0.6*'+y_pred_list[3][0]+' + 0.4*'+y_pred_list[4][0], metrics.mean_squared_error(y_test, y_pred_blend).round(4))
    dff = pd.DataFrame({'datetime': time_test[:, 0], 'mnth': time_test[:, 1], 'hr': time_test[:, 2], 'cnt': np.expm1(y_test), 'prediction': y_pred_blend})
    dff.to_csv('../output/prediction_blended.csv', index = False, columns=['datetime', 'mnth', 'hr', 'cnt', 'prediction'])
    print('blended predictions saved in ../output/prediction_blended.csv')
    df.to_csv('../data/hour_ext.csv')
    print('extended dataset saved in ../data/hour_ext.csv')
        
def grid_search_est(fn, features, est, param, outfn):
    """ hyperparameter tuning for models using GridSearchCV
        Parameters
        ----------
        fn: file name of dataset
        features: feature set for different models
        est: sklearn estimator / regressor
        param: set of parameters to be tuned
        outfn: file name for storing predication made by the model with best tuned params
        
        Returns
        -------
        best_params_: best tuned params
    """
    est_name = str(est).split('(')[0]
    print('grid_search_est --', est_name)
    df = load_data(fn)
    check_df(df)
    df = feature_engineering(df)
    df, X_train, X_test, y_train, y_test, y_train_cas, y_test_cas, y_train_reg, y_test_reg, time_test = split_data(df, features=features)

    # this may take a while .....
    gs_cv = GridSearchCV(est, param, n_jobs=4, verbose=2)
    gs_cv.fit(X_train, y_train)
     
    result = gs_cv.predict(X_test)
    mse_test = metrics.mean_squared_error(y_test, result)
    print('grid_search_est --', est_name, '-- best params: ', gs_cv.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in gs_cv.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print('grid_search_est --', est_name, '-- mse on test set: ', mse_test)
    result = np.expm1(result)
    dff = pd.DataFrame({'datetime': time_test[:, 0], 'mnth': time_test[:, 1], 'hr': time_test[:, 2], 'cnt': np.expm1(y_test), 'prediction': result})
    dff.to_csv(outfn, index = False, columns=['datetime', 'mnth', 'hr', 'cnt', 'prediction'])
    print('grid_search_est --', est_name, '-- predictions saved in', outfn)
    
    return gs_cv.best_params_
        
if __name__ == "__main__":
#    features = [['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'day', 'month', 'year', 'dow', 'woy', 'peak_hr', 'atemp_windspeed', 'hr_cat', 'temp_cat', 'hum_cat', 'cnt_season'], ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'day', 'month', 'year', 'dow', 'woy', 'peak_hr', 'atemp_windspeed', 'hr_cat', 'temp_cat', 'hum_cat', 'cnt_season'], ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'day', 'month', 'year', 'dow', 'woy', 'peak_hr', 'atemp_windspeed', 'hr_cat', 'temp_cat', 'hum_cat', 'cnt_season'], ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'day', 'month', 'year', 'dow', 'woy', 'peak_hr', 'atemp_windspeed', 'hr_cat', 'temp_cat', 'hum_cat', 'cnt_season'], ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']] #featureset1
    features = [['hr', 'temp', 'peak_hr', 'dow', 'workingday', 'year', 'hum', 'woy', 'weathersit', 'season', 'atemp', 'temp_cat', 'yr'], ['hr', 'hr_cat', 'peak_hr', 'workingday', 'temp_cat', 'cnt_season', 'temp', 'atemp', 'yr', 'weathersit', 'year', 'season', 'hum_cat', 'hum', 'dow', 'woy', 'weekday', 'mnth', 'windspeed'], ['hr', 'hr_cat', 'temp', 'peak_hr', 'workingday', 'dow', 'atemp', 'woy', 'hum', 'year', 'yr', 'weathersit', 'season', 'day', 'windspeed', 'cnt_season', 'weekday', 'temp_cat'], ['hr', 'woy', 'day', 'hum', 'dow', 'hr_cat', 'atemp', 'temp', 'workingday', 'windspeed', 'weekday', 'weathersit', 'peak_hr', 'holiday', 'year', 'month', 'yr', 'season', 'mnth', 'cnt_season'], ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]#featureset2
    """ GridSearchCV
    param = {'learning_rate': [0.1, 0.05],
             'max_depth': [5, 10, 15],
             'min_samples_leaf': [5, 10, 20],
             }
    est = GradientBoostingRegressor(n_estimators=150)
    grid_search_est(fn='../data/hour.csv', features=features[3], est=est, param=param, outfn='../output/prediction_gs_gbm.csv')
    param = {'C': [1, 10, 20, 30, 40]
             }
    est = svm.SVR()
    grid_search_est(fn='../data/hour.csv', features=features[4], est=est, param=param, outfn='../output/prediction_gs_svr.csv')
    param = {'n_estimators': [500, 1000],
             'max_depth': [5, 10, 15],
             'min_samples_split': [3, 5, 10],
             }
    est = RandomForestRegressor(n_jobs=-1)
    grid_search_est(fn='../data/hour.csv', features=features[2], est=est, param=param, outfn='../output/prediction_gs_rfr.csv')
    """
    test_run(fn='../data/hour.csv', features=features, type='featureset2')
    
""" current running result:
test_run featureset2
Index(['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt', 'day', 'month', 'year', 'dow', 'woy',
       'peak_hr', 'atemp_windspeed', 'hr_cat', 'temp_cat', 'hum_cat'],
      dtype='object')
featureset2 DecisionTreeRegressor 0.1448
featureset2 ExtraTreesRegressor 0.0976
featureset2 RandomForestRegressor 0.0953
featureset2 GradientBoostingRegressor 0.0743
featureset2 SVR 0.0809
featureset2 blending: 0.2*RandomForestRegressor + 0.8*GradientBoostingRegressor 0.0755
featureset2 blending: 0.3*ExtraTreesRegressor + 0.7*GradientBoostingRegressor 0.0752
featureset2 blending: 0.2*GradientBoostingRegressor + 0.8*SVR 0.073
featureset2 blending: 0.6*GradientBoostingRegressor + 0.4*SVR 0.0699
blended predictions saved in ../output/prediction_blended.csv
extended dataset saved in ../data/hour_ext.csv"""
