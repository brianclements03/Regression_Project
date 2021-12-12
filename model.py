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




