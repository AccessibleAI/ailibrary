# Linear Regression

## General
In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.
This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.

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
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html



 

