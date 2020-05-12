"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Dec 18, 2019
Updated by: Omer Liberman

xgb.py
==============================================================================
"""
import argparse
import pandas as pd

from SKTrainer import *
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def _cast_types(args):
	"""
	This method performs casting to all types of inputs passed via cmd.
	:param args: argparse.ArgumentParser object.
	:return: argparse.ArgumentParser object.
	"""
	args.x_val = None if args.x_val == 'None' else int(args.x_val)
	args.test_size = float(args.test_size)
	args.max_depth = int(args.max_depth)
	args.learning_rate = float(args.learning_rate)
	args.n_estimators = int(args.n_estimators)
	args.verbosity = int(args.verbosity)
	# objective.
	# booster.
	# tree_method.
	args.n_jobs = int(args.n_jobs)
	args.gamma = float(args.gamma)
	args.min_child_weight = int(args.min_child_weight)
	args.max_delta_step = int(args.max_delta_step)
	args.subsample = int(args.subsample)
	args.colsample_bytree = float(args.colsample_bytree)
	args.colsample_bylevel = float(args.colsample_bylevel)
	args.colsample_bynode = float(args.colsample_bynode)
	args.reg_alpha = float(args.reg_alpha)
	args.reg_lambda = float(args.reg_lambda)
	args.scale_pos_weight = float(args.scale_pos_weight)
	args.base_score = float(args.base_score)
	args.random_state = int(args.random_state)
	args.missing = None if args.missing == 'None' else float(args.missing)

	return args


def main(args):
	args = _cast_types(args)

	# Loading data set.
	data = pd.read_csv(args.data, index_col=0)
	for col in data.columns:
		if col.startswith('Unnamed'):
			data = data.drop(columns=col, axis=1)

	# Checking data sets sizes.
	rows_num, cols_num = data.shape
	if rows_num == 0:
		raise Exception("Library Error: The given dataset has no examples.")
	if cols_num < 2:
		raise Exception("Dataset Error: Not enough columns.")

	# Split to X and y.
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

	# Model initialization.
	model = XGBClassifier(
		max_depth=args.max_depth,
		learning_rate=args.learning_rate,
		n_estimators=args.n_estimators,
		verbosity=args.verbosity,
		objective=args.objective,
		booster=args.booster,
		tree_method=args.tree_method,
		n_jobs=args.n_jobs,
		gamma=args.gamma,
		min_child_weight=args.min_child_weight,
		max_delta_step=args.max_delta_step,
		subsample=args.subsample,
		colsample_bytree=args.colsample_bytree,
		colsample_bylevel=args.colsample_bylevel,
		colsample_bynode=args.colsample_bynode,
		reg_alpha=args.reg_alpha,
		reg_lambda=args.reg_lambda,
		scale_pos_weight=args.scale_pos_weight,
		base_score=args.base_score,
		random_state=args.random_state,
		missing=args.missing)

	folds = None if args.x_val is None else args.x_val

	trainer = SKTrainer(model=model,
						train_set=(X_train, y_train),
						test_set=(X_test, y_test),
						output_model_name=args.output_model,
						testing_mode=args.test_mode,
						folds=folds)

	trainer.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""xgboost Classifier""")

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
	parser.add_argument('--max_depth', action='store', default="3", dest='max_depth',
						help="""(int) – Maximum tree depth for base learners. .Default is 3""")

	parser.add_argument('--learning_rate', action='store', default="0.1", dest='learning_rate',
						help="""(float) – Boosting learning rate (xgb’s “eta”) .Default is 0.1""")

	parser.add_argument('--n_estimators', action='store', default="100", dest='n_estimators',
						help="""(int) – Number of trees to fit. Default is 100""")

	parser.add_argument('--verbosity', action='store', default="1", dest='verbosity',
						help="""(int) – The degree of verbosity. Valid values are 0 (silent) - 3 (debug). Default is 1""")

	parser.add_argument('--objective', action='store', default='binary:logistic', dest='objective',
						help=""": (string or callable) – Specify the learning task and the corresponding learning objective or a custom objective function to 
						be used (see note below). . Default is 'binary:logistic'""")

	parser.add_argument('--booster', action='store', default='gbtree', dest='booster',
						help="""(string) – Specify which booster to use: gbtree, gblinear or dart.
						 Default is 'gbtree'""")

	parser.add_argument('--tree_method', action='store', default='auto', dest='tree_method',
						help="""(string) – Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. 
						It’s recommended to study this option from parameters document.""")

	parser.add_argument('--n_jobs', action='store', default="1", dest='n_jobs',
						help="""(int) – Number of parallel threads used to run xgboost. . Default is 1""")

	parser.add_argument('--gamma', action='store', default="0", dest='gamma',
						help="""(float) – Minimum loss reduction required to make a further partition on a leaf node of the tree. . Default is 0""")

	parser.add_argument('--min_child_weight', action='store', default="1", dest='min_child_weight',
						help="""min_child_weight (int) – Minimum sum of instance weight(hessian) needed in a child. . Default is 1""")

	parser.add_argument('--max_delta_step', action='store', default="0", dest='max_delta_step',
						help="""(int) – Maximum delta step we allow each tree’s weight estimation to be. . Default is 0""")

	parser.add_argument('--subsample', action='store', default="1", dest='subsample',
						help="""(float) – Subsample ratio of the training instance. Default is 1""")

	parser.add_argument('--colsample_bytree', action='store', default="1", dest='colsample_bytree',
						help="""(float) – Subsample ratio of columns when constructing each tree. . Default is 1""")

	parser.add_argument('--colsample_bylevel', action='store', default="1", dest='colsample_bylevel',
						help="""(float) – Subsample ratio of columns for each level. Default is 1""")

	parser.add_argument('--colsample_bynode', action='store', default="1", dest='colsample_bynode',
						help="""(float) – Subsample ratio of columns for each split. Default is 1""")

	parser.add_argument('--reg_alpha', action='store', default="0", dest='reg_alpha',
						help="""(float (xgb's alpha)) – L1 regularization term on weights. Default is 0""")

	parser.add_argument('--reg_lambda', action='store', default="1", dest='reg_lambda',
						help="""(float (xgb's lambda)) – L2 regularization term on weights. Default is 1""")

	parser.add_argument('--scale_pos_weight', action='store', default="1", dest='scale_pos_weight',
						help="""(float) – Balancing of positive and negative weights. Default is 1""")

	parser.add_argument('--base_score', action='store', default="0.5", dest='base_score',
						help="""The initial prediction score of all instances, global bias. . Default is 0.5""")

	parser.add_argument('--random_state', action='store', default="0", dest='random_state',
						help=""" (int) – Random number seed.. Default is 0""")

	parser.add_argument('--missing', action='store', default="None", dest='missing',
						help="""(float, optional) – Value in the data which needs to be present as a missing value. If None, defaults to
						 np.nan. . Default is None""")

	args = parser.parse_args()

	main(args)
