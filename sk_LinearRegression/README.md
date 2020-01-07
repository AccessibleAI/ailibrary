# Linear Regression

## General

In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). When one explanatory variable is used it is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.
This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.

## Notes for this library

1) The library enables the use of the algorithm both with cross validation and without. By default the library doesn't perform cross validation. If the user wishes to perform cross validation, 
the user needs to use the parameter: ```--x_val=NUMBER_OF_FOLDS```, which is ```--x_val=None``` by default.  
2) The path given by ```--data``` must be a path to a csv file which is already processed and ready for training. This means that the csv must not contain: 
   - NaN values (empty cells) 
   - Strings 
   - Columns whose names start with 'Unnamed'.
  
## Parameters


### cnvrg.io parameters

```--data``` - str, required. Path to `.csv` file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table is built as such: all the columns but the 
rightmost one are considered as features columns (x), and the rightmost one is the label column (y).

```--x_val``` - int, optional (default = 5). Number of folds for the cross-validation.

```--test_split``` - float, optional (default = 0.2). The portion of the data for testing.

```--output_model``` - str, optional (default = 'GradientBoostModel.sav') The name of the output file which is a trained model. 


### algorithm parameters

```--fit_intercept``` - boolean, optional (default = True). Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).

```--normalize``` - boolean, optional (default = False). This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.

```--copy_X``` - boolean, optional (default = True). If True, X will be copied. Otherwise, it may be overwritten.

```--n_jobs``` - int or None, optional (default = None). The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficiently large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html



 

