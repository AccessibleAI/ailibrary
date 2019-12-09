"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Oct 06, 2019
Updated by: Omer Liberman

gradient_boosting.py
==============================================================================
"""
import argparse
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from cnvrg_sklearn_helper import train_with_cross_validation, train_without_cross_validation

import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def _cast_types(args):
	"""
	This method performs casting to all types of inputs passed via cmd.
	:param args: argparse.ArgumentParser object.
	:return: argparse.ArgumentParser object.
	"""
	# x_val.
	if args.x_val != 'None':
		args.x_val = int(args.x_val)
	else:
		args.x_val = None

	# test_size
	args.test_size = float(args.test_size)

	# n_neighbors.
	args.n_neighbors = int(args.n_neighbors)

	# leaf_size.
	args.leaf_size = int(args.leaf_size)

	# p.
	args.p = int(args.p)

	# metric_params.
	if args.metric_params == "None" or args.metric_params == 'None':
		args.metric_params = None

	# n_jobs.
	if args.n_jobs == "None" or args.n_jobs == 'None':
		args.n_jobs = None
	else:
		args.n_jobs = int(args.n_jobs)
	#  --- ---------------------------------------- --- #
	return args


def main(args):
	args = _cast_types(args)

	# Loading dataset.
	data = pd.read_csv(args.data)
	for col in data.columns:
		if col.startswith('Unnamed'):
			data = data.drop(columns=col, axis=1)

	# Checking data sets sizes.
	rows_num, cols_num = data.shape
	if rows_num == 0:
		raise Exception("Dataset Error: The given dataset has no examples.")
	if cols_num < 2:
		raise Exception("Dataset Error: Not enough columns.")

	X = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

	model = GradientBoostingClassifier(
		loss=args.loss,
		learning_rate=args.learning_rate,
		n_estimators=args.n_estimators,
		subsample=args.subsample,
		criterion=args.criterion,
		min_samples_split=args.min_samples_split,
		min_samples_leaf=args.min_samples_leaf,
		min_weight_fraction_leaf=args.min_weight_fraction_leaf,
		max_depth=args.max_depth,
		min_impurity_decrease=args.min_impurity_decrease,
		min_impurity_split=args.min_impurity_split,
		init=args.init,
		random_state=args.random_state,
		max_features=args.max_features,
		verbose=args.verbose,
		max_leaf_nodes=args.max_leaf_nodes,
		warm_start=args.warm_start,
		presort=args.presort,
		validation_fraction=args.validation_fraction,
		n_iter_no_change=args.n_iter_no_change,
		tol=args.tol
	)

	# Training with cross validation.
	if args.x_val is not None:
		train_with_cross_validation(model=model,
									train_set=(X_train, y_train),
									test_set=(X_test, y_test),
									folds=args.x_val,
									project_dir=args.project_dir,
									output_model_name=args.output_model,
									testing_mode=args.test_mode)

	# Training without cross validation.
	else:
		train_without_cross_validation(model=model,
										train_set=(X_train, y_train),
										test_set=(X_test, y_test),
										project_dir=args.project_dir,
										output_model_name=args.output_model,
										testing_mode=args.test_mode)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""K-Nearest-Neighbors Classifier""")
	# ----- cnvrg.io params.
	parser.add_argument('--data', action='store', dest='data', required=True,
	                    help="""String. path to csv file: The data set for the classifier. Assumes the last column includes the labels. """)

	parser.add_argument('--project_dir', action='store', dest='project_dir',
	                    help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_dir', action='store', dest='output_dir',
	                    help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--x_val', action='store', default="None", dest='x_val',
	                    help="""Integer. Number of folds for the cross-validation. Default is None.""")

	parser.add_argument('--test_size', action='store', default="0.2", dest='test_size',
	                    help="""Float. The portion of the data of testing. Default is 0.2""")

	parser.add_argument('--output_model', action='store', default="model.sav", dest='output_model',
	                    help="""String. The name of the output file which is a trained random forests model """)

	parser.add_argument('--test_mode', action='store', default=False, dest='test_mode',
						help="""--- For inner use of cnvrg.io ---""")

	# ----- model's params.
	parser.add_argument('--loss', action='store', default="deviance", dest='loss',
	                    help="""{‘deviance’, ‘exponential’}, optional (default=’deviance’) loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.""")

	parser.add_argument('--learning_rate', action='store', default="0.1", dest='learning_rate',
	                    help="""float, optional (default=0.1) learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.""")

	parser.add_argument('--n_estimators', action='store', default="100", dest='n_estimators',
	                    help="""int (default=100) The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.""")

	parser.add_argument('--subsample', action='store', default="1.0", dest='subsample',
	                    help="""float, optional (default=1.0) The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.""")

	parser.add_argument('--criterion', action='store', default="friedman_mse", dest='criterion',
	                    help="""string, optional (default=”friedman_mse”) The function to measure the quality of a split. Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.""")

	parser.add_argument('--min_samples_split', action='store', default="2", dest='min_samples_split',
						help="""int, float, optional (default=2) The minimum number of samples required to split an internal node:
						If int, then consider min_samples_split as the minimum number.
						If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.""")

	parser.add_argument('--min_samples_leaf', action='store', default="1", dest='min_samples_leaf',
						help="""int, float, optional (default=1) The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
						If int, then consider min_samples_leaf as the minimum number.
						If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.""")

	parser.add_argument('--min_weight_fraction_leaf', action='store', default="0.", dest='min_weight_fraction_leaf',
						help="""float, optional (default=0.) The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. """)

	parser.add_argument('--max_depth', action='store', default="3", dest='max_depth',
						help="""integer, optional (default=3) maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.""")

	parser.add_argument('--min_impurity_decrease', action='store', default="0.", dest='min_impurity_decrease',
						help="""float, optional (default=0.) A node will be split if this split induces a decrease of the impurity greater than or equal to this value.""")

	parser.add_argument('--min_impurity_split', action='store', default="1e-7", dest='min_impurity_split',
						help="""float, (default=1e-7) Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.""")

	parser.add_argument('--init', action='store', default="None", dest='init',
						help="""estimator or ‘zero’, optional (default=None) An estimator object that is used to compute the initial predictions. init has to provide fit and predict_proba. If ‘zero’, the initial raw predictions are set to zero. By default, a DummyEstimator predicting the classes priors is used. """)

	parser.add_argument('--random_state', action='store', default="None", dest='random_state',
						help="""int, RandomState instance or None, optional (default=None) If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. """)

	parser.add_argument('--max_features', action='store', default="None", dest='max_features',
						help="""int, float, string or None, optional (default=None) The number of features to consider when looking for the best split:
						If int, then consider max_features features at each split.
						If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
						If “auto”, then max_features=sqrt(n_features).
						If “sqrt”, then max_features=sqrt(n_features).
						If “log2”, then max_features=log2(n_features).
						If None, then max_features=n_features.
						Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
						Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. """)

	parser.add_argument('--verbose', action='store', default="0", dest='verbose',
						help="""int, default: 0. Enable verbose output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree. """)

	parser.add_argument('--max_leaf_nodes', action='store', default="None", dest='max_leaf_nodes',
						help="""int or None, optional (default=None) Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.""")

	parser.add_argument('--warm_start', action='store', default="False", dest='warm_start',
						help="""bool, default: False. When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution""")

	parser.add_argument('--presort', action='store', default="deprecated", dest='presort',
						help="""deprecated, default=’deprecated’. This parameter is deprecated and will be removed in v0.24. """)

	parser.add_argument('--validation_fraction', action='store', default="0.1", dest='validation_fraction',
						help="""float, optional, default 0.1. The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.""")

	parser.add_argument('--n_iter_no_change', action='store', default="None", dest='n_iter_no_change',
						help="""int, default None. n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations. The split is stratified. """)

	parser.add_argument('--tol', action='store', default="1e-4", dest='tol',
						help="""float, optional, default 1e-4. Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops. """)

	args = parser.parse_args()

	main(args)
