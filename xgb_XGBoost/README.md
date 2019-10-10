# XGBoost

## General
XGBoost is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, and Julia. It works on Linux, Windows, and macOS.
From the project description, it aims to provide a scalable, portable and distributed gradient boosting library.

## Note for this library
The library enables to use the algorithm both with cross validation and without. By default the library doesn't perform cross validation. If the user wishes to perform cross validation, 
the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```, which is ```--x_val=None``` by default.
*** 
The library uses the XGBClassifier (```from xgboost import XGBClassifier```).
***

## Parameters
### cnvrg.io params
```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table built like: all the columns but the 
rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

```--output_model``` - String. The name of the output file which is a trained model. Default is xgboost_model.sav .

### algorithm params
https://xgboost.readthedocs.io/en/latest/parameter.html