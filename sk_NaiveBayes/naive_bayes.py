"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Michal Ettudgi

Last update: Oct 06, 2019
Updated by: Omer Liberman

logistic_regression.py
==============================================================================
"""
import argparse
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from cnvrg_sklearn_helper import train_with_cross_validation, train_without_cross_validation


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
	# alpha
	args.alpha = float(args.alpha)

	# fit_prior
	args.fit_prior = (args.fit_prior in ['True', "True", 'true', "true"])

	# class_prior - array like type (problem to convert)
	if args.class_prior == "None" or args.class_prior == 'None':
		args.class_prior = None

	return args


def main(args):
	args = _cast_types(args)

	# Loading data, and splitting it to train and test based on user input
	data = pd.read_csv(args.data)
	for col in data.columns:
		if col.startswith('Unnamed'):
			data = data.drop(columns=col, axis=1)

	# Check for unfit given dataset and splitting to X and y.
	rows_num, cols_num = data.shape
	if rows_num == 0:
		raise Exception("Library Error: The given dataset has no examples.")
	if cols_num < 2:
		raise Exception("Dataset Error: Not enough columns.")

	X = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

	# Initializing classifier with user input
	model = MultinomialNB(alpha=args.alpha,
	                      fit_prior=args.fit_prior,
	                      class_prior=args.class_prior)

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
	parser = argparse.ArgumentParser(description="""MultinomialNB""")

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
	                    help="""String. The name of the output file which is a trained random forests model. Default is logistic_regression_model.sav""")

	parser.add_argument('--test_mode', action='store', default=False, dest='test_mode',
						help="""--- For inner use of cnvrg.io ---""")

	# ----- model's params.
	parser.add_argument('--alpha', action='store', default="0.1", dest='alpha',
	                    help="""float: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)""")

	parser.add_argument('--fit_prior',  action='store', default="True", dest='fit_prior',
	                    help="""boolean Whether to learn class prior probabilities or not. If false, a uniform prior will be used.""")

	parser.add_argument('--class_prior', action='store', default=None, dest='class_prior',
	                    help="""Prior probabilities of the classes. If specified the priors are not adjusted according to the data.""")

	args = parser.parse_args()
	main(args)



