An implementation of Knn for classification. In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.

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

### algorithm parameters

```--n_neighbors``` - int, optional (default = 5). Number of neighbors to use by default for kneighbors queries.

```--weights``` - str or callable, optional (default = 'uniform'). Weight function used in prediction. Possible values:  
'uniform' : uniform weights. All points in each neighborhood are weighted equally.  
'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.  
[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.  

```--algorithm``` - {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default = 'auto'). Algorithm used to compute the nearest neighbors:  
'ball_tree' will use BallTree.  
'kd_tree' will use KDTree.  
'brute' will use a brute-force search.  
'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method.  
Note: fitting on sparse input will override the setting of this parameter, using brute force.

```--leaf_size``` - int, optional (default = 30). Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. 
The optimal value depends on the nature of the problem.

```--p``` - integer, optional (default = 2). Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

```--metric``` - string or callable (default = 'minkowski'). The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.

```--metric_params``` - dict, optional (default = None). Additional keyword arguments for the metric function.

```--n_jobs``` - int or None, optional (default = None). The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. Doesn't affect fit method.

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html