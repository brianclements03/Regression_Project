import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")
# importing my personal wrangle module
import wrangle


def create_data_for_models(X_train_scaled, X_validate_scaled, X_test_scaled):
    '''
    This function takes a DataFrame and manipulates it (usually by dropping features) to arrive at a set of features
    to put into different models.  For instance, the X_train_kbest is a scaled DataFrame based on X_train_scaled, with
    all features dropped except those given by my kbest feature engineering function in previous cells.
    
    '''
    X_train_kbest = X_train_scaled.drop(columns = ['bedrooms', 'sq_ft_per_bathroom', 'LA',
       'Orange', 'Ventura'])
    X_validate_kbest = X_validate_scaled.drop(columns = ['bedrooms', 'sq_ft_per_bathroom', 'LA',
       'Orange', 'Ventura'])
    X_test_kbest = X_test_scaled.drop(columns = ['bedrooms', 'sq_ft_per_bathroom', 'LA',
       'Orange', 'Ventura'])
    X_train_rfe = X_train_scaled.drop(columns = ['bedrooms', 'bathrooms', 'age', 'sq_ft_per_bathroom', 'Ventura'])
    X_validate_rfe = X_validate_scaled.drop(columns = ['bedrooms', 'bathrooms', 'age', 'sq_ft_per_bathroom', 'Ventura'])
    X_test_rfe = X_test_scaled.drop(columns = ['bedrooms', 'bathrooms', 'age', 'sq_ft_per_bathroom', 'Ventura'])

    return X_train_kbest, X_validate_kbest, X_test_kbest, X_train_rfe, X_validate_rfe, X_test_rfe




def run_ols_model_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df):
    from sklearn.metrics import mean_squared_error
    # create the model object
    lm = LinearRegression()
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_kbest, y_train_scaled.tax_value)
    # predict train
    y_train_scaled['tax_value_pred_lm_kbest'] = lm.predict(X_train_kbest)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lm_kbest) ** .5
    # predict validate
    y_validate_scaled['tax_value_pred_lm_kbest'] = lm.predict(X_validate_kbest)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lm_kbest) ** (0.5)
    
    metric_df = metric_df.append({
        'model': 'OLS Regressor KBEST', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)

    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
    #     "\nValidation/Out-of-Sample: ", rmse_validate)

    return metric_df


def run_ols_model_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df):
    from sklearn.metrics import mean_squared_error
    # create the model object
    lm = LinearRegression()
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_rfe, y_train_scaled.tax_value)
    # predict train
    y_train_scaled['tax_value_pred_lm_rfe'] = lm.predict(X_train_rfe)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lm_rfe) ** .5
    # predict validate
    y_validate_scaled['tax_value_pred_lm_rfe'] = lm.predict(X_validate_rfe)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lm_rfe) ** (0.5)
    
    metric_df = metric_df.append({
        'model': 'OLS Regressor RFE', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)

    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
    #     "\nValidation/Out-of-Sample: ", rmse_validate)

    return metric_df


def lasso_lars_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df):

    # a good balance is a low rmse and a low difference

    lars = LassoLars(alpha= 1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train_kbest, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_lars_kbest'] = lars.predict(X_train_kbest)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lars_kbest) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_lars_kbest'] = lars.predict(X_validate_kbest)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lars_kbest) ** (1/2)
    metric_df = metric_df.append({
        'model': 'Lasso_alpha1_KBEST', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    
    return metric_df


def lasso_lars_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df):


    # THIS IS LARS WITH RFE
    lars = LassoLars(alpha= .1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train_rfe, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_lars_rfe'] = lars.predict(X_train_rfe)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lars_rfe) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_lars_rfe'] = lars.predict(X_validate_rfe)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lars_rfe) ** (1/2)

    metric_df = metric_df.append({
    'model': 'Lasso_alpha1_RFE', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    
    return metric_df



def tweedie_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df):

# as seen in curriculum, the power ought to be set per distribution type
# power = 0 is same as OLS

    glm = TweedieRegressor(power=1.4, alpha=0)


    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_kbest, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_glm_kbest'] = glm.predict(X_train_kbest)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_glm_kbest) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_glm_kbest'] = glm.predict(X_validate_kbest)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_glm_kbest) ** (1/2)

    metric_df = metric_df.append({
        'model': 'glm_compound_kbest', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        }, ignore_index=True)
    return metric_df

def tweedie_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df):
        # Tweedie on RFE features:

    # as seen in curriculum, the power ought to be set per distribution type
    # power = 0 is same as OLS

    glm = TweedieRegressor(power=1.5, alpha=0)


    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_rfe, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_glm_rfe'] = glm.predict(X_train_rfe)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_glm_rfe) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_glm_rfe'] = glm.predict(X_validate_rfe)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_glm_rfe) ** (1/2)

    metric_df = metric_df.append({
    'model': 'glm_compound_rfe', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    return metric_df


def polynomial_regression_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, X_test_kbest, metric_df):
        # make the polynomial features to get a new set of features. import from sklearn
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree2_kbest = pf.fit_transform(X_train_kbest)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2_kbest = pf.transform(X_validate_kbest)
    X_test_degree2_kbest =  pf.transform(X_test_kbest)
    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2_kbest, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_lm2_kbest'] = lm2.predict(X_train_degree2_kbest)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lm2_kbest) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_lm2_kbest'] = lm2.predict(X_validate_degree2_kbest)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lm2_kbest) ** 0.5

    metric_df = metric_df.append({
    'model': 'quadratic_kbest', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    return metric_df


def polynomial_regression_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, X_test_rfe, metric_df):

    # make the polynomial features to get a new set of features. import from sklearn
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2_rfe = pf.fit_transform(X_train_rfe)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2_rfe = pf.transform(X_validate_rfe)
    X_test_degree2_rfe =  pf.transform(X_test_rfe)

    # rfe features here:


    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2_rfe, y_train_scaled.tax_value)

    # predict train
    y_train_scaled['tax_value_pred_lm2_rfe'] = lm2.predict(X_train_degree2_rfe)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train_scaled.tax_value, y_train_scaled.tax_value_pred_lm2_rfe) ** (1/2)

    # predict validate
    y_validate_scaled['tax_value_pred_lm2_rfe'] = lm2.predict(X_validate_degree2_rfe)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate_scaled.tax_value, y_validate_scaled.tax_value_pred_lm2_rfe) ** 0.5

    metric_df = metric_df.append({
    'model': 'quadratic_rfe', 
    'RMSE_train': rmse_train,
    'RMSE_validate': rmse_validate,
    }, ignore_index=True)
    return metric_df


def run_all_models_on_all_data(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, X_train_rfe, X_validate_rfe, X_test_kbest, X_test_rfe, metric_df):
    metric_df = run_ols_model_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df)
    metric_df = run_ols_model_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df)
    metric_df = lasso_lars_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df)
    metric_df = lasso_lars_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df)
    metric_df = tweedie_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, metric_df)
    metric_df = tweedie_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, metric_df)
    metric_df = polynomial_regression_kbest(X_train_kbest, y_train_scaled, X_validate_kbest, y_validate_scaled, \
                                              X_test_kbest, metric_df)
    metric_df = polynomial_regression_rfe(X_train_rfe, y_train_scaled, X_validate_rfe, y_validate_scaled, X_test_rfe, metric_df)

    return metric_df