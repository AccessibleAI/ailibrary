# Gradient Boosting

## General

Gradient Boosting is used for classification. It builds an additive model in a forward stage-wise fashion and it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 

Binary classification is a special case where only a single regression tree is induced.

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

```--x_val``` - int, optional (default = 5). Number of folds for the cross-validation.

```--test_split``` - float, optional (default = 0.2). The portion of the data for testing.

```--output_model``` - str, optional (default = 'GradientBoostModel.sav') The name of the output file which is a trained model. 


### algorithm parameters

```--loss``` - {‘deviance’, ‘exponential’}, optional (default = ’deviance’). Loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
 
```--learning_rate``` - float, optional (default = 0.1). Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.

```--n_estimators``` - int (default = 100). The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

```--subsample``` - float, optional (default = 1.0). The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0, this results in Stochastic Gradient Boosting. Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.

```--criterion``` - str, optional (default = ”friedman_mse”). The function to measure the quality of a split. Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.

```--min_samples_split``` - int, float, optional (default = 2). The minimum number of samples required to split an internal node:
 - If int, then consider min_samples_split as the minimum number.
 - If float, then min_samples_split is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.

```--min_samples_leaf``` - int, float, optional (default = 1). The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
 - If int, then consider min_samples_leaf as the minimum number.
 - If float, then min_samples_leaf is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

```--min_weight_fraction_leaf``` - float, optional (default = 0). The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

```--max_depth``` - integer, optional (default = 3). The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.

```--min_impurity_decrease``` - float, optional (default = 0). A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

```--min_impurity_split``` - float (default = 1e-7). Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

```--init``` - estimator or ‘zero’, optional (default = None). An estimator object that is used to compute the initial predictions. init has to provide fit and predict_proba. If ‘zero’, the initial raw predictions are set to zero. By default, a DummyEstimator predicting the classes priors is used.

```--random_state``` - int, RandomState instance or None, optional (default = None). 
 - If int, random_state is the seed used by the random number generator.
 - If RandomState instance, random_state is the random number generator.
 - If None, the random number generator is the RandomState instance used by np.random.

```--max_features``` - int, float, str or None, optional (default = None). The number of features to consider when looking for the best split:
 - If int, then consider max_features features at each split.
 - If float, then max_features is a fraction and `int(max_features * n_features)` features are considered at each split.
 - If “auto”, then `max_features=sqrt(n_features)`.
 - If “sqrt”, then `max_features=sqrt(n_features)` (same as “auto”).
 - If “log2”, then `max_features=log2(n_features)`.
 - If None, then `max_features=n_features`.
Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

```--verbose``` - int (default = 0). Enable verbose output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree.

```--max_leaf_nodes``` - int or None, optional (default = None). Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None, then unlimited number of leaf nodes.

```--warm_start``` - bool (default = False). When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.

```--presort``` - deprecated (default = ’deprecated’). This parameter is deprecated and will be removed in the future.

```--validation_fraction``` - float, optional (default =  0.1). The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.

```--n_iter_no_change``` - int (default = None). n_iter_no_change is used to decide if early stopping will be used to terminate training when the validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations. The split is stratified.

```--tol``` - float, optional (default = 1e-4). Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html