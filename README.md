# REGRESSION

## This file/repo contains information related to my Regression project, using thee Zillow dataset from the Codeup database.

## The Plan

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the Codeup database using a MySQL query; manipulating the data to a form suitable to the exploration of variables and machine learning models to be applied; and to graphically present the most outstanding findings, including actionable conclusions, along the way.

##  Steps to Reproduce

In  the case of this project, there are several python files that can be used to acquire, clean, prepare and otherwise manipulate the data in advance of exploration, feature selection, and modeling (listed below).

I split the data into X_train and y_train data sets for much of the exploration and modelling, and was careful that no features were directly dependent on the target variable, tax_value.  I created a couple of features of my own, which helped me determine features to use, and dropped rows with null values (my final dataset was 44,864 rows long, down from 52,442 that were downloaded using SQL)

Once the data is correctly prepared, it can be run through the sklearn preprocessing feature for polynomial regressiong and fit on the scaled X_train dataset, using only those features indicated from the recursive polynomial engineering feature selector (also an sklearn function).  This provided me with the best results for the purposes of my project.

LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- explore.py: contains a few functions that were helpful exploring the data.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions

## Project Goals

The ultimate goal of this project is to build a model that predicts the tax value of the homes in question with a higher accuracy than the baseline I have chosen. I will use Residual Mean Square Error as my metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.

The final deliverable will be the RMSE value resulting from my best model, contrasted with the baseline RMSE.

Additionally, a Jupyter Notebook with my main findings and conclusions will be a key deliverable; many .py files will exist as a back-up to the main Notebook (think "under-the-hood" coding that will facilitate the presentation).

## Project Description

This Jupyter Notebook and presentation explore the Zillow dataset from the Codeup database. The data used relates to 2017 real estate transactions relating to single family homes in three California counties, and different aspects of the properties. An important aspect of the Zillow business model is to be able to publish accurate home values; I intend to build a machine learning model that makes predictions about the tax value of the homes in question.

## Data Dictionary

Variable	Meaning
___________________
bedrooms:	The number of bedrooms
bathrooms:	The number of bathrooms
sq_ft:	How many square feet
tax_value:	The tax value of the home
county:	What county does it belong to
age:	How old is it?
sq_ft_per_bathroom:	How many square feet per bathroom?
LA:	Belongs to Los Angeles county
Orange:	Belongs to Orange county
Ventura:	Belings to Ventura county

Variables created in the notebook (explanation where it helps for clarity):

train
validate
test
X_train
y_train
X_validate
y_validate
X_test
y_test
train_scaled
X_train_scaled
y_train_scaled
validate_scaled
X_validate_scaled
y_validate_scaled
test_scaled
X_test_scaled
y_test_scaled



## Initial Questions

- Are larger homes valued higher?  What about older homes?

- Do more bathrooms relate to higher tax value? What about square feet per bathroom--is there a sweet spot?

- Newer homes are larger; they are also valued more highly. Is there an exception to the rule?


## Key findings, recommendations and takeaways

- I have found that ...




# NOTES ON YOUR README
- review the reproduce section, not all those modules existed yet as of the time you wrote it!
- what is the final goal? a dataframe of predictions against actual tax values? or just the rmse as comparted to baseline?
- what about your key findings (etc) section?



