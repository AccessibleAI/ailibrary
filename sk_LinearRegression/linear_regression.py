"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Oct 15, 2019
Updated by: Omer Liberman

logistic_regression.py
==============================================================================
"""
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from regression_helper import train_with_cross_validation, train_without_cross_validation


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

	# test_size.
	args.test_size = float(args.test_size)

	# fit_intercept.
	args.fit_intercept = (args.fit_intercept in ['True', "True", 'true', "true"])

	# normalize.
	args.normalize = (args.normalize in ['True', "True", 'true', "true"])

	# copy_X.
	args.copy_X = (args.copy_X in ['True', "True", 'true', "true"])

	# n_jobs.
	if args.n_jobs == 'None':
		args.n_jobs = None
	else:
		args.n_jobs = int(args.n_jobs)

	# --------------- #
	return args


def main(args):
	args = _cast_types(args)

	# Loading data as df, and splitting it to train and test based on user input
	data = pd.read_csv(args.data)
	for col in data.columns:
		if col.startswith('Unnamed'):
			data = data.drop(columns=col, axis=1)

	rows_num, cols_num = data.shape

	if rows_num == 0:
		raise Exception("Dataset Error: The given dataset has no examples.")
	if cols_num < 2:
		raise Exception("Dataset Error: Not enough columns.")

	X = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

	# Initializing model with user input
	model = LinearRegression(fit_intercept=args.fit_intercept,
	                         normalize=args.normalize,
	                         copy_X=args.copy_X,
	                         n_jobs=args.n_jobs)

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
	parser = argparse.ArgumentParser(description="""linear regression""")
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
	                    help="""String. The name of the output file which is a trained model. Default is linear_regression_model.sav""")

	parser.add_argument('--test_mode', action='store', default=False, dest='test_mode',
						help="""--- For inner use of cnvrg.io ---""")

	# ----- model's params.
	parser.add_argument('--fit_intercept', action='store', default='True', dest='fit_intercept',
	                    help="""boolean, optional, default True. whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).""")

	parser.add_argument('--normalize', action='store', default='False', dest='normalize',
	                    help="""boolean, optional, default False. This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.""")

	parser.add_argument('--copy_X', action='store', default='True', dest='copy_X',
	                    help="""boolean, optional, default True. If True, X will be copied; else, it may be overwritten.""")

	parser.add_argument('--n_jobs', action='store', default='None', dest='n_jobs',
	                    help="""int or None, optional (default=None). The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.""")

	args = parser.parse_args()

	main(args)