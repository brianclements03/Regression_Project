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

The ultimate goal of this project is to build a model that predicts the tax value of the homes in question with a higher accuracy than the baseline I have chosen. I will use Residual Mean Square Error as my metric for evaluation.

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


## Initial Questions

- Are larger homes valued higher?  What about older homes?

- Do more bathrooms relate to higher tax value? What about square feet per bathroom--is there a sweet spot?

- Newer homes are larger; they are also valued more highly. Is there an exception to the rule?

# NOTES ON YOUR README
- review the reproduce section, not all those modules existed yet as of the time you wrote it!




