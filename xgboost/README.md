# XGBoost

## General
XGBoost for classification.
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
```--max_depth``` - (int) – Maximum tree depth for base learners.

```--learning_rate``` - (float) – Boosting learning rate (xgb’s “eta”)

```--n_estimators``` - (int) – Number of trees to fit.

```--verbosity``` - (int) – The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

```--objective``` - (string or callable) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).

```--booster``` - (string) – Specify which booster to use: gbtree, gblinear or dart.

```--tree_method``` - (string) – Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. It’s recommended to study this option from parameters document.

```--n_jobs``` - (int) – Number of parallel threads used to run xgboost.

```--gamma``` - (float) – Minimum loss reduction required to make a further partition on a leaf node of the tree.

```--min_child_weight``` - (int) – Minimum sum of instance weight(hessian) needed in a child.

```--max_delta_step``` - (int) – Maximum delta step we allow each tree’s weight estimation to be.

```--subsample``` - (float) – Subsample ratio of the training instance.

```--colsample_bytree``` - (float) – Subsample ratio of columns when constructing each tree.

```--colsample_bylevel``` - (float) – Subsample ratio of columns for each level.

```--colsample_bynode``` - (float) – Subsample ratio of columns for each split.

```--reg_alpha``` - (float (xgb's alpha)) – L1 regularization term on weights.

```--reg_lambda``` - (float (xgb's lambda)) – L2 regularization term on weights.

```--scale_pos_weight``` - (float) – Balancing of positive and negative weights.

```--base_score``` - The initial prediction score of all instances, global bias.

```--random_state``` - (int) – Random number seed.

```--missing``` - (float, optional) – Value in the data which needs to be present as a missing value. If None, defaults to np.nan.

## Link
https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn