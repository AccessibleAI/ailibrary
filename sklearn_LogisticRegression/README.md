# Logistic Regression

## General
In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing 
such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such 
as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned 
a probability between 0 and 1 and the sum adding to one.

## Note for this library
The library enables to use the algorithm both with cross validation and without.
By default the library doesn't perform cross validation.
If the user wishes to perform cross validation, the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```,
which is ```--x_val=None``` by default.

## Parameters

### cnvrg.io params

* ```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and 
the table built like: all the columns but the rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

* ```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

* ```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

* ```--output_model``` - String. The name of the output file which is a trained random forests model. Default is RandomForestModel.sav

### algorithm params
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html




 

