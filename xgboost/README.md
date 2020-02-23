XGBoost for classification.
XGBoost is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, and Julia. It works on Linux, Windows, and macOS.
From the project description, it aims to provide a scalable, portable and distributed gradient boosting library.

## Notes for this Component
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

```--test_size``` - Float. The portion of the data of testing. Default is 0.2.


```--x_val``` - int, optional (default = 5). Number of folds for the cross-validation.

```--test_split``` - float, optional (default = 0.2). The portion of the data for testing.

```--output_model``` - str, optional (default = 'xgboost_model.sav') The name of the output file which is a trained model. 

### Algorithm parameters

```--max_depth``` - int (default = 3). The maximum tree depth for base learners.

```--learning_rate``` - float (default = 0.1). Boosting learning rate (xgb’s “eta”).

```--n_estimators``` - int (default = 100). Number of trees to fit.

```--verbosity``` - int (default = 0). The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

```--objective``` - str or callable (default = binary:logistic). Specify the learning task and the corresponding learning objective or a custom objective function to be used.

```--booster``` - str (default = 'gbtree'). Specify which booster to use: 'gbtree', 'gblinear' or 'dart'.

```--tree_method``` - str (default = 'auto'). Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. Go to the xgboost documentation at the link below for more information.

```--n_jobs``` - int (default = '1'). Number of parallel threads used to run xgboost.

```--gamma``` - float (default = '0'). Minimum loss reduction required to make a further partition on a leaf node of the tree.

```--min_child_weight``` - int (default = '1'). Minimum sum of instance weight(hessian) needed in a child.

```--max_delta_step``` - int (default = '0'). Maximum delta step to allow each tree’s weight estimation to be.

```--subsample``` - float (default = '1'). Subsample ratio of the training instance.

```--colsample_bytree``` - float (default = '1'). Subsample ratio of columns when constructing each tree.

```--colsample_bylevel``` - float (default = '1'). Subsample ratio of columns for each level.

```--colsample_bynode``` - float (default = '1'). Subsample ratio of columns for each split.

```--reg_alpha``` - float (default = '0'). xgb's alpha. L1 regularization term on weights.

```--reg_lambda``` - float (default = '1'). xgb's lambda. L2 regularization term on weights.

```--scale_pos_weight``` - float (default = '1'). Balancing of positive and negative weights.

```--base_score``` - float (default = '0.5'). The initial prediction score of all instances, global bias.

```--random_state``` - int (default = '0'). Random number seed.

```--missing``` - float, optional (default = None). Value in the data which needs to be present as a missing value. If None, defaults to np.nan.

## Link
https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn