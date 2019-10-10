"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Oct 06, 2019
Updated by: Omer Liberman

knn.py
==============================================================================
"""
import argparse
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
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

	model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
	                             weights=args.weights,
	                             algorithm=args.algorithm,
	                             leaf_size=args.leaf_size,
	                             p=args.p,
	                             metric=args.metric,
	                             metric_params=args.metric_params,
	                             n_jobs=args.n_jobs)

	# Training with cross validation.
	if args.x_val is not None:
		train_with_cross_validation(model=model,
		                            train_set=(X_train, y_train),
		                            test_set=(X_test, y_test),
		                            folds=args.x_val,
		                            project_dir=args.project_dir,
		                            output_model_name=args.output_model)

	# Training without cross validation.
	else:
		train_without_cross_validation(model=model,
		                               train_set=(X_train, y_train),
		                               test_set=(X_test, y_test),
		                               project_dir=args.project_dir,
		                               output_model_name=args.output_model)



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

	parser.add_argument('--output_model', action='store', default="kNearestNeighborsModel.sav", dest='output_model',
	                    help="""String. The name of the output file which is a trained random forests model """)

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

	main(args)
