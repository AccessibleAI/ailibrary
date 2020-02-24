Naive Bayes classifier for multinomial models. The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The
multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

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

```--output_model``` - str, optional (default = 'kNearestNeighborsModel.sav') The name of the output file which is a trained model. 

### Algorithm parameters

```--alpha``` - float, optional (default = 1.0). Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

```--fit_prior``` - bool, optional (default = True). Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

```--class_prior``` - array-like, size (n_classes), optional (default = None). Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html