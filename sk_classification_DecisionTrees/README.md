Decision Trees Classifier is a simple and widely used classification technique. It applies a straight forward idea to solve the classification problem. Decision Trees Classifier poses a series of carefully crafted questions about the attributes of the test record. Each time it receives an answer, a follow-up question is asked until a conclusion about the class label of the record is reached.

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

```--output_model``` - str, optional (default = 'decisionTreesClassifierModel.sav') The name of the output file which is a trained model. 

### Algorithm parameters

```--criterion``` - str, optional (default = 'gini'). The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

```--splitter``` - str, optional (default = 'best'). The strategy used to choose the split at each node. Supported strategies are 'best' to choose the best split and 'random' to choose the best random split.

```--max_depth``` - int or None, optional (default = None). The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

```--min_samples_split``` - int or float, optional (default = 2). The minimum number of samples required to split an internal node:  
 - If int, then consider min_samples_split as the minimum number.  
 - If float, then min_samples_split is a fraction and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
                            
```--min_samples_leaf``` - int or float, optional (default = 1). The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at 
least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.  
 - If int, then consider min_samples_leaf as the minimum number.  
 - If float, then min_samples_leaf is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

```--min_weight_fraction_leaf``` -  float, optional (default = 0). The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

```--max_features``` - int, float, str or None, optional (default = None).
The number of features to consider when looking for the best split:  
 - If int, then consider max_features features at each split.  
 - If float, then max_features is a fraction and `int(max_features * n_features)` features are considered at each split.  
 - If 'auto', then `max_features=sqrt(n_features)`.  
 - If 'sqrt', then `max_features=sqrt(n_features)` (same as “auto”).
 - If 'log2', then `max_features=log2(n_features)`.  
 - If None, then `max_features=n_features`.  
Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

```--random_state``` - int, RandomState instance or None, optional (default = None).  
 - If int, random_state is the seed used by the random number generator.   
 - If RandomState instance, random_state is the 
random number generator.  
 - If None, the random number generator is the RandomState instance used by np.random.

```--max_leaf_nodes``` - int or None, optional (default = None). Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None, then unlimited number of leaf nodes.

```--min_impurity_decrease``` - float, optional (default = 0).
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
The weighted impurity decrease equation is the following:  
N_t / N * (impurity - N_t_R / N_t * right_impurity  
                    - N_t_L / N_t * left_impurity)  
where `N` is the total number of samples, `N_t` is the number of samples at the current node, `N_t_L` is the number of samples in the left child, and `N_t_R` is the number of samples in the right child.
`N`, `N_t`, `N_t_R` and `N_t_L` all refer to the weighted sum, if sample_weight is passed.

```--min_impurity_split``` - float, (default = 1e-7). Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

```--class_weight``` - dict, list of dicts, 'balanced' or None, (default = None). Weights associated with classes in the form {class_label: weight}.   
If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.  
Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].  
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
For multi-output, the weights of each column of y will be multiplied.

```--presort``` - bool, optional (default = False). Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. When using either a smaller dataset or a restricted depth, this may speed up the training.


## Link
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html