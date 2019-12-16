# MultinomialNB (Naive Bayes)

## General
Naive Bayes classifier for multinomial models
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The
multinomial distribution normally requires integer feature counts. However,in practice, fractional counts such as tf-idf may also work.

## Note for this library
The library enables to use the algorithm both with cross validation and without. By default the library doesn't perform cross validation. If the user wishes to perform cross validation, 
the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```, which is ```--x_val=None``` by default.

## Parameters
### cnvrg.io params
```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table built like: all the columns but the 
rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

```--output_model``` - String. The name of the output file which is a trained model. Default is naive_bayes_model.sav

### algorithm params
```--alpha``` - float, optional (default=1.0) Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

```--fit_prior``` - boolean, optional (default=True). Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

```--class_prior``` - array-like, size (n_classes,), optional (default=None). Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html