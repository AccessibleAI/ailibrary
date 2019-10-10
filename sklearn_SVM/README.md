# SVM classifier (Support Vector Machine)

## General
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). 
An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. 
New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.

## Note for this library
The library enables to use the algorithm both with cross validation and without. By default the library doesn't perform cross validation. 
If the user wishes to perform cross validation,  the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```, 
which is ```--x_val=None``` by default.
*** 
The library uses the SVC classifier (```sklearn.svm.SVC```).
***

## Parameters
### cnvrg.io params
```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table built like: all the columns but the 
rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

```--output_model``` - String. The name of the output file which is a trained model. Default is svm_model.sav .

### algorithm params
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html