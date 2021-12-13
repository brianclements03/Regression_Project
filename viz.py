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

def actual_vs_predicted(y_train_scaled):
    # plot to visualize actual vs predicted. 
    plt.hist(y_train_scaled.tax_value_pred_mean, color='red', alpha=.5,  label="Predicted Tax Values - Mean")
    plt.hist(y_train_scaled.tax_value, color='blue', alpha=.5, label="Actual Tax Values")
    #plt.hist(y_train.G3_pred_median, bins=1, color='orange', alpha=.5, label="Predicted Final Grades - Median")
    plt.xlabel("Tax Value")
    plt.ylabel("Number of Properties")
    plt.legend()
    plt.show()


def age_by_county(train):
    plt.figure(figsize = (16,3))
    plt.subplot(1,3, 1)

    # Title with column name.
    plt.title('LA County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='LA'].age)
    # Hide gridlines.

    plt.subplot(1,3, 2)
    # Title with column name.
    plt.title('Orange County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='Orange'].age)
    # Hide gridlines.

    plt.subplot(1,3, 3)
    # Title with column name.
    plt.title('Ventura County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='Ventura'].age)
    # Hide gridlines.

    plt.grid(False)
    plt.tight_layout()