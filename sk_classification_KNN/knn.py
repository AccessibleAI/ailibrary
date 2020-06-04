"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Jun 01, 2020
Updated by: Omer Liberman

knn.py
==============================================================================
"""
import argparse
from sklearn.neighbors import KNeighborsClassifier
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
	args.n_neighbors = int(args.n_neighbors)
	args.leaf_size = int(args.leaf_size)
	args.p = int(args.p)

	# metric_params.
	if args.metric_params == "None" or args.metric_params == 'None':
		args.metric_params = None

	args.n_jobs = None if args.n_jobs == 'None' else int(args.n_jobs)
	#  --- ------------- --- #
	return args


def _parse_arguments():
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

	parser.add_argument('--train_loss_type', action='store', default='MSE', dest='train_loss_type',
						help='(string) (default: MSE) can be one of: F1, LOG, MSE, RMSE, MAE, R2.')

	parser.add_argument('--test_loss_type', action='store', default='MSE', dest='test_loss_type',
						help='(string) (default: MSE) can be one of: F1, LOG, MSE, RMSE, MAE, R2, zero_one_loss.')

	parser.add_argument('--digits_to_round', action='store', default='4', dest='digits_to_round',
						help="""(int) (default: 4) the number of decimal numbers to round.""")

	parser.add_argument('--output_model', action='store', default="model.sav", dest='output_model_name',
						help="""String. The name of the output file which is a trained model """)

	parser.add_argument('--test_mode', action='store', default=False, dest='test_mode',
						help="""--- For inner use of cnvrg.io ---""")

	# ----- model's params.
	parser.add_argument('--n_neighbors', action='store', default="5", dest='n_neighbors',
						help=""" Number of neighbors to use by default for kneighbors queries. .Default is 5""")

	parser.add_argument('--weights', action='store', default='uniform', dest='weights',
						help=""" weight function used in prediction. Possible values:
	                        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
	                        ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
	                        [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
	                        .Default is 3""")

	parser.add_argument('--algorithm', action='store', default='auto', dest='algorithm',
						help=""" Algorithm used to compute the nearest neighbors:
	                        ‘ball_tree’ will use BallTree
	                        ‘kd_tree’ will use KDTree
	                        ‘brute’ will use a brute-force search.
	                        ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit 
	                        method.""")

	parser.add_argument('--leaf_size', action='store', default="30", dest='leaf_size',
						help=""" Leaf size passed to BallTree or KDTree. This can affect the speed of the construction 
	                        and query, as well as the memory required to store the tree. The optimal value depends on the 
	                        nature of the problem. .Default is 30""")

	parser.add_argument('--p', action='store', default="2", dest='p',
						help=""" Power parameter for the Minkowski metric. When p = 1, this is equivalent to using 
	                        manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, 
	                        minkowski_distance (l_p) is used. .Default is 2""")

	parser.add_argument('--metric', action='store', default='minkowski', dest='metric',
						help=""": the distance metric to use for the tree. The default metric is minkowski, and with 
	                        p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric
	                        class for a list of available metrics . Default is ‘minkowski’""")

	# dictionary
	parser.add_argument('--metric_params', action='store', default="None", dest='metric_params',
						help=""": Additional keyword arguments for the metric function.. Default is None""")

	parser.add_argument('--n_jobs', action='store', default="1", dest='n_jobs',
						help=""": --- . Default is 1""")

	args = parser.parse_args()
	return args


def main(args):
	args = _cast_types(args)

	# Initializing classifier with user input
	model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
	                             weights=args.weights,
	                             algorithm=args.algorithm,
	                             leaf_size=args.leaf_size,
	                             p=args.p,
	                             metric=args.metric,
	                             metric_params=args.metric_params,
	                             n_jobs=args.n_jobs)

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
