# Decision Trees Classifier

## General
Decision Tree Classifier is a simple and widely used classification technique. 
It applies a straitforward idea to solve the classification problem. Decision Tree Classifier poses a series of carefully crafted questions about the attributes of the test record. 
Each time time it receive an answer,a follow-up question is asked until a conclusion about the calss label of the record is reached.

## Note for this library
The library enables to use the algorithm both with cross validation and without. By default the library doesn't perform cross validation. If the user wishes to perform cross validation, 
the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```, which is ```--x_val=None``` by default.

## Parameters
### cnvrg.io params
```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table built like: all the columns but the 
rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

```--output_model``` - String. The name of the output file which is a trained model. Default is decision_trees_model.sav

### algorithm params
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html