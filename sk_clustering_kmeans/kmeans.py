"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Yishai Raswosky

Last update: Oct 06, 2019
Updated by: Omer Liberman

knn.py
==============================================================================
"""
import pickle
import argparse
import pandas as pd

from cnvrg import Experiment
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from SKTrainerClustering import SKTrainerClustering

def _cast_types(args):
	args.folds = int(args.folds)
	args.n_clusters = int(args.n_clusters)
	args.n_init = int(args.n_init)
	args.max_iter = int(args.max_iter)
	args.tol = float(args.tol)

	if args.precompute_distances == "auto": pass
	elif args.precompute_distances == "True": args.precompute_distances = True
	elif args.precompute_distances == "False": args.precompute_distances = False
	else: pass

	args.verbose = int(args.verbose)
	args.random_state = None if args.random_state == 'None' else int(args.random_state)
	args.copy_x = bool(args.copy_x == 'True')
	args.n_jobs = None if args.n_jobs == 'None' else int(args.n_jobs)

	if args.algorithm in ['auto', 'full', 'elkan']: pass
	else: raise ValueError("ParameterError: Algorithm setting not recognized.")

	return args

def main(args):
	args = _cast_types(args)

	# Minimal number of rows and columns in the csv file.
	MINIMAL_NUM_OF_ROWS = 10
	MINIMAL_NUM_OF_COLUMNS = 2

	# Loading data, and splitting it to train and test based on user input
	data = pd.read_csv(args.data, index_col=0)

	# Check for unfit given dataset and splitting to X and y.
	rows_num, cols_num = data.shape
	if rows_num < MINIMAL_NUM_OF_ROWS: raise ValueError("LibraryError: The given csv doesn't have enough rows (at least 10 examples must be given).")
	if cols_num < MINIMAL_NUM_OF_COLUMNS: raise ValueError("DatasetError: Not enough columns in the csv (at least 2 columns must be given).")

	X = data
	X_train, X_test = train_test_split(X, test_size=args.test_size)

	# Initializing classifier with user input
	model = KMeans(init=args.init,
				   n_init=args.n_init,
				   n_clusters=args.n_clusters,
				   max_iter=args.max_iter,
				   tol=args.tol,
				   precompute_distances=args.precompute_distances,
				   verbose=args.verbose,
				   random_state=args.random_state,
				   copy_x=args.copy_x,
				   n_jobs=args.n_jobs,
				   algorithm=args.algorithm)

	trainer = SKTrainerClustering(model=model,
								  train_set=X_train,
								  test_set=X_test,
								  output_model_name=args.output_model,
								  testing_mode=args.test_mode)
    
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""sklearn_KMeans (not prepared)""")

	parser.add_argument('--data',action='store',required=True,dest='data',
						help="""A path to the dataset. It should be a path to .csv file where the rightmost column is the target column and the other are the X. All columns should be numbers.""")

	parser.add_argument('--project_dir', action='store', dest='project_dir',
						help="""String.. """)

	parser.add_argument('--output_dir', action='store', dest='output_dir',
						help="""String.. """)

	parser.add_argument('--model',action='store',default="K-MeansModel.sav",dest='model',
						help="""String. The name of the output file which is a trained random forests model.""")

	parser.add_argument('--folds', action='store', default="5", dest='folds',
						help="""Integer. Number of folds for the cross-validation. Default is 5.""")

	parser.add_argument('--n_clusters',action='store',default="8",dest='n_clusters',
						help="""The number of clusters to form as well as the number of centroids to generate.""")

	parser.add_argument('--init',action='store',default='k-means++',dest='init',
						help="""Method for initialization, defaults to ‘k-means++’: ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details. ‘random’: choose k observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.""")

	parser.add_argument('--n_init',action='store',default='10',dest='n_init',
						help="""Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.""")

	parser.add_argument('--max_iter',action='store',default='300',dest='max_iter',
						help="""Maximum number of iterations of the k-means algorithm for a single run.""")

	parser.add_argument('--tol',action='store',default='1e-4',dest='tol',
						help="""Relative tolerance with regards to inertia to declare convergence.""")

	parser.add_argument('--precompute_distances',action='store',default='auto',dest='precompute_distances',
						help="""Precompute distances (faster but takes more memory). ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision. True : always precompute distances. False : never precompute distances.""")

	parser.add_argument('--verbose',action='store',default='0',dest='verbose',
						help="""Verbosity mode.""")

	parser.add_argument('--random_state',action='store',default='None',dest='random_state',
						help="""Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.""")

	parser.add_argument('--copy_x',action='store',default='True',dest='copy_x',
						help="""When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified, ensuring X is C-contiguous. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean, in this case it will also not ensure that data is C-contiguous which may cause a significant slowdown.""")

	parser.add_argument('--n_jobs',action='store',default='None',dest='n_jobs',
						help="""The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.""")

	parser.add_argument('--algorithm',action='store',default='auto',dest='algorithm',
						help="""K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data.""")

	args = parser.parse_args()

	main(args)
