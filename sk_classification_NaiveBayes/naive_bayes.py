"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Michal Ettudgi

Last update: Jun 01, 2020
Updated by: Omer Liberman

naive_bayes.py
==============================================================================
"""
import argparse
from sklearn.naive_bayes import MultinomialNB
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
	args.alpha = float(args.alpha)
	args.fit_prior = (args.fit_prior == 'True')

	# class_prior - array like type (problem to convert)
	if args.class_prior == "None" or args.class_prior == 'None':
		args.class_prior = None
	return args


def _parse_arguments():
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
	parser.add_argument('--alpha', action='store', default="0.1", dest='alpha',
	                    help="""float: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)""")

	parser.add_argument('--fit_prior',  action='store', default="True", dest='fit_prior',
	                    help="""boolean Whether to learn class prior probabilities or not. If false, a uniform prior will be used.""")

	parser.add_argument('--class_prior', action='store', default=None, dest='class_prior',
	                    help="""Prior probabilities of the classes. If specified the priors are not adjusted according to the data.""")

	args = parser.parse_args()
	return args


def main(args):
	args = _cast_types(args)

	# Initializing classifier with user input
	model = MultinomialNB(alpha=args.alpha,
	                      fit_prior=args.fit_prior,
	                      class_prior=args.class_prior)

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



