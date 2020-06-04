"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Michal Ettudgi

Last update: Jun 01, 2020
Updated by: Omer Liberman

svm.py
==============================================================================
"""
import argparse
from sklearn import svm
from utils.scikit_learn.sk_trainer import SKTrainerClassification


def _cast_types(args):
	"""
	This method performs casting to all types of inputs passed via cmd.
	:param args: argparse.ArgumentParser object.
	:return: argparse.ArgumentParser object.
	"""
	args.x_val = None if args.x_val == 'None' else int(args.x_val)
	args.test_size = float(args.test_size)
	args.digits_to_round = int(args.digits_to_round)
	args.C = float(args.C)
	# kernel
	args.degree = int(args.degree)
	# gamma
	args.coef0 = float(args.coef0)
	args.shrinking = (args.shrinking == 'True')
	args.probability = (args.probability == 'True')
	args.tol = float(args.tol)
	args.cache_size = float(args.cache_size)

	# class_weight
	if args.class_weight == "None" or args.class_weight == 'None':
		args.class_weight = None

	args.verbose = (args.verbose == 'True')
	args.max_iter = int(args.max_iter)
	# decision_function_shape
	args.random_state = None if args.random_state == 'None' else int(args.random_state)

	return args


def _parse_arguments():
	parser = argparse.ArgumentParser(description="""sk_classification_SVM""")
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
	parser.add_argument('--C', action='store', default="1.0", dest='C',
						help="""Penalty parameter C of the error term.""")

	parser.add_argument('--kernel', action='store', default="rbf", dest='kernel',
						help="""Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
	                         ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the 
	                         kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).""")

	parser.add_argument('--degree', action='store', default="3", dest='degree',
						help="""Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.""")

	parser.add_argument('--gamma', action='store', default='auto', dest='gamma',
						help="""Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) 
	                        as value of gamma. The current default of gamma, ‘auto’, will change to 
	                        ‘scale’ in version 0.22. ‘auto_deprecated’, a deprecated version of ‘auto’ is used as a default indicating that no explicit
	                        value of gamma was passed.""")

	parser.add_argument('--coef0', action='store', default="0.0", dest='coef0',
						help="""Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.""")

	parser.add_argument('--shrinking', action='store', default="True", dest='shrinking',
						help="""Whether to use the shrinking heuristic.""")

	parser.add_argument('--probability', action='store', default="False", dest='probability',
						help=""" Whether to enable probability estimates. This must be enabled prior to calling fit, 
	                        and will slow down that method.""")

	parser.add_argument('--tol', action='store', default="1e-3", dest='tol',
						help="""Tolerance for stopping criterion.""")

	parser.add_argument('--cache_size', action='store', default="200.0", dest='cache_size',
						help="""Specify the size of the kernel cache (in MB).""")

	parser.add_argument('--class_weight', default='None', action='store', dest='class_weight',
						help="""Set the parameter C of class i to class_weight[i]*C for SVC.
	                         If not given, all classes are supposed to have weight one.
	                          The “balanced” mode uses the values of y to automatically adjust weights inversely proportional 
	                          to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))""")

	parser.add_argument('--verbose', action='store', default="False", dest='verbose',
						help="""Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that,
	                         if enabled, may not work properly in a multithreaded context.""")

	parser.add_argument('--max_iter', action='store', default="-1", dest='max_iter',
						help="""Hard limit on iterations within solver, or -1 for no limit.""")

	parser.add_argument('--decision_function_shape', action='store', default="ovr", dest='decision_function_shape',
						help="""Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers,
	                         or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). 
	                        However, one-vs-one (‘ovo’) is always used as multi-class strategy.""")

	parser.add_argument('--random_state', action='store', default='None', dest='random_state',
						help="""The seed of the pseudo random number generator used when shuffling the data for probability estimates.
	                         If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; 
	                        If None, the random number generator is the RandomState instance used by np.random""")

	args = parser.parse_args()
	return args


def main(args):
	args = _cast_types(args)

	# Initializing classifier with user input
	model = svm.SVC(
		C=args.C,
		kernel=args.kernel,
		degree=args.degree,
		gamma=args.gamma,
		coef0=args.coef0,
		shrinking=args.shrinking,
		probability=args.probability,
		tol=args.tol,
		cache_size=args.cache_size,
		class_weight=args.class_weight,
		verbose=args.verbose,
		max_iter=args.max_iter,
		decision_function_shape=args.decision_function_shape,
		random_state=args.random_state)

	folds = None if args.x_val is None else args.x_val

	trainer = SKTrainerClassification(sk_learn_model_object=model,
									  path_to_csv_file=args.data,
									  test_size=args.test_size,
									  output_model_name=args.output_model_name,
									  train_loss_type=args.train_loss_type,
									  test_loss_type=args.test_loss_type,
									  digits_to_round=args.digits_to_round,
									  folds=folds)

	trainer.run()


if __name__ == '__main__':
	args = _parse_arguments()
	main(args)



