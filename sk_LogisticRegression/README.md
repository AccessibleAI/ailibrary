# Logistic Regression

## General
In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. 
This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a 
probability between 0 and 1 and the sum adding to one.

## Note for this library
The library enables to use the algorithm both with cross validation and without.
By default the library doesn't perform cross validation.
If the user wishes to perform cross validation, the user need to use the parameter: ```--x_val=NUMBER_OF_FOLDS```,
which is ```--x_val=None``` by default.

## Parameters

### cnvrg.io params

* ```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and 
the table built like: all the columns but the rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

* ```--x_val``` - Integer. Number of folds for the cross-validation. Default is 5.

* ```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

* ```--output_model``` - String. The name of the output file which is a trained random forests model. Default is RandomForestModel.sav

### algorithm params
* ```--penalty``` - str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
					Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.

* ```--dual``` - bool, optional (default=False) Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
				 Prefer dual=False when n_samples > n_features.

* ```tol``` - float, optional (default=1e-4) Tolerance for stopping criteria.

* ```--C``` - float, optional (default=1.0) Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

* ```--fit_intercept``` - bool, optional (default=True) Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

* ```--intercept_scaling``` - float, optional (default=1) Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling],
	                          i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. 
	                          The intercept becomes intercept_scaling * synthetic_feature_weight.

* ```--class_weight``` - dict or ‘balanced’, optional (default=None) Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have 
	                     weight one.

* ```--random_state``` - int, RandomState instance or None, optional (default=None). The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the
	                     seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState 
	                     instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.

* ```--solver``` - str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’lbfgs’)
                   Algorithm to use in the optimization problem.
                   For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
                   For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
                   ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
                   ‘liblinear’ and ‘saga’ also handle L1 penalty
                   ‘saga’ also supports ‘elasticnet’ penalty
                   ‘liblinear’ does not support setting penalty='none'

* ```--max_iter``` - int, optional (default=100) Maximum number of iterations taken for the solvers to converge. 

* ```--multi_class``` - {‘ovr’, ‘multinomial’, ‘auto’}, default=’auto’ If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is 
	                    the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ 
	                    if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’. 

* ```---verbose``` - int, optional (default=0). For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

* ```--warm_start``` - bool, optional (default=False). When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
	                   Useless for liblinear solver.  

* ```n_jobs``` - int or None, optional (default=None)
				 Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the solver is set to ‘liblinear’ regardless 
				 of whether ‘multi_class’ is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

* ```--l1_ratio``` - float or None, optional (default=None). The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'`. Setting 
	                 ``l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a
	                 combination of L1 and L2.

## Link
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


 

