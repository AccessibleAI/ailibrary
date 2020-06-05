"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Jun 01, 2020
Updated by: Omer Liberman

linear_regression.py
==============================================================================
"""
import argparse
from sklearn.linear_model import LinearRegression
from utils.scikit_learn.regression.sk_trainer_regression import SKTrainerRegression


def _cast_types(args):
	"""
	This method performs casting to all types of inputs passed via cmd.
	:param args: argparse.ArgumentParser object.
	:return: argparse.ArgumentParser object.
	"""
	args.x_val = None if args.x_val == 'None' else int(args.x_val)
	args.test_size = float(args.test_size)
	args.digits_to_round = int(args.digits_to_round)
	args.fit_intercept = (args.fit_intercept == 'True')
	args.normalize = (args.normalize == 'True')
	args.copy_X = (args.copy_X == 'True')
	args.n_jobs = None if args.n_jobs == 'None' else int(args.n_jobs)
	# --------------- #
	return args


def _parse_arguments():
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

	parser.add_argument('--train_loss_type', action='store', default='MSE', dest='train_loss_type',
						help='(string) (default: MSE) can be one of: F1, LOG, MSE, RMSE, MAE, R2.')

	parser.add_argument('--test_loss_type', action='store', default='MSE', dest='test_loss_type',
						help='(string) (default: MSE) can be one of: F1, LOG, MSE, RMSE, MAE, R2, zero_one_loss.')

	parser.add_argument('--digits_to_round', action='store', default='4', dest='digits_to_round',
						help="""(int) (default: 4) the number of decimal numbers to round.""")

	parser.add_argument('--output_model', action='store', default="model.sav", dest='output_model_name',
						help="""String. The name of the output file which is a trained model. Default is model.sav""")

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
	return args


def main(args):
	args = _cast_types(args)

	# Initializing model with user input
	model = LinearRegression(fit_intercept=args.fit_intercept,
	                         normalize=args.normalize,
	                         copy_X=args.copy_X,
	                         n_jobs=args.n_jobs)

	trainer = SKTrainerRegression(sk_learn_model_object=model,
									  path_to_csv_file=args.data,
									  test_size=args.test_size,
									  output_model_name=args.output_model_name,
									  train_loss_type=args.train_loss_type,
									  test_loss_type=args.test_loss_type,
									  digits_to_round=args.digits_to_round,
									  folds=args.x_val)

	trainer.run()


if __name__ == '__main__':
	args = _parse_arguments()
	main(args)