# SVM Classifier (Support Vector Machine)

## General

An implementation of SVM for classification. In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 

Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). 

An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.

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

```--output_model``` - str, optional (default = 'svm_model.sav') The name of the output file which is a trained model. 

### algorithm parameters

```--C``` - float, optional (default = 1.0). Penalty parameter C of the error term.

```--kernel``` - str, optional (default = ’rbf’). Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

```--degree``` - int, optional (default = 3). Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

```--gamma``` - float, optional (default = ’auto’). Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

```--coef0``` - float, optional (default = 0.0). Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

```--shrinking``` - bool, optional (default = True). Whether to use the shrinking heuristic.

```--probability``` - bool, optional (default = False). Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.

```--tol``` - float, optional (default = 1e-3). Tolerance for stopping criterion.

```--cache_size``` - float, optional (default = '200.0'). Specify the size of the kernel cache (in MB).

```--class_weight``` - dict, ‘balanced’ or None, optional (default = None). Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`.

```--verbose``` - bool, (default = False). Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

```--max_iter``` - int, optional (default = -1). Hard limit on iterations within solver, or -1 for no limit.

```--decision_function_shape``` - ‘ovo’, ‘ovr’ (default = ’ovr’). Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy.

```--random_state``` - int, RandomState instance or None, optional (default = None). The seed of the pseudo random number generator to use when shuffling the data. 
 - If int, random_state is the seed used by the random number generator.
 - If RandomState instance, random_state is the random number generator.
 - If None, the random number generator is the RandomState instance used by np.random. 

## Links
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html