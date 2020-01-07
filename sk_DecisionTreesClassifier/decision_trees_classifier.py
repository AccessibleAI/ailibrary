"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Michal Ettudgi

Last update: Oct 06, 2019
Updated by: Omer Liberman

decision_trees_classifier.py
==============================================================================
"""
import argparse
import pandas as pd

from SKTrainer import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def _cast_types(args):
	"""
	This method performs casting to all types of inputs passed via cmd.
	:param args: argparse.ArgumentParser object.
	:return: argparse.ArgumentParser object.
	"""
	args.x_val = int(args.x_val) if args.x_val != 'None' else None
	args.test_size = float(args.test_size)
	# criterion (string)
	# splitter (string)
	args.max_depth = None if args.max_depth == 'None' else float(args.max_depth)
	args.min_samples_split = int(args.min_samples_split)
	args.min_samples_leaf = int(args.min_samples_leaf)
	args.min_weight_fraction_leaf = float(args.min_weight_fraction_leaf)
	args.max_features = None if args.max_features == 'None' else float(args.max_features)
	args.random_state = None if args.random_state == 'None' else float(args.random_state)
	args.max_leaf_nodes = None if args.max_leaf_nodes == 'None' else float(args.max_leaf_nodes)
	args.min_impurity_decrease = float(args.min_impurity_decrease)
	args.class_weight = None if args.class_weight == 'None' else float(args.class_weight)
	args.presort = (args.presort in ['True', 'true'] )
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
	model = DecisionTreeClassifier(
		criterion=args.criterion,
		splitter=args.splitter,
		max_depth=args.max_depth,
		min_samples_split=args.min_samples_split,
		min_samples_leaf=args.min_samples_leaf,
		min_weight_fraction_leaf=args.min_weight_fraction_leaf,
		max_features=args.max_features,
		random_state=args.random_state,
		max_leaf_nodes=args.max_leaf_nodes,
		min_impurity_decrease=args.min_impurity_decrease,
		class_weight=args.class_weight,
		presort=args.presort)

	folds = None if args.x_val is None else args.x_val

	trainer = SKTrainer(model=model,
						train_set=(X_train, y_train),
						test_set=(X_test, y_test),
						output_model_name=args.output_model,
						testing_mode=args.test_mode,
						folds=folds)

	trainer.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""DecisionTreeClassifier""")
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
	parser.add_argument('--criterion', action='store', default="gini", dest='criterion',
	                    help="""The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.""")

	parser.add_argument('--splitter', action='store', default="best", dest='splitter',
	                    help="""The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.""")

	parser.add_argument('--max_depth' , action='store', default='None', dest='max_depth',
	                    help="""The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.""")

	parser.add_argument('--min_samples_split',  action='store', default="2", dest='min_samples_split',
	                    help="""The minimum number of samples required to split an internal node""")

	parser.add_argument('--min_samples_leaf', action='store', default="1", dest='min_samples_leaf',
	                    help="""The minimum number of samples required to be at a leaf node.""")

	parser.add_argument('--min_weight_fraction_leaf', action='store', default="0.", dest='min_weight_fraction_leaf',
	                    help="""The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
                        Samples have equal weight when sample_weight is not provided.""")

	parser.add_argument('--max_features', action='store', default='None', dest='max_features',
	                    help="""The number of features to consider when looking for the best split""")

	parser.add_argument('--random_state',action='store', default='None', dest='random_state',
	                    help="""If int, random_state is the seed used by the random number generator; 
                        If RandomState instance, random_state is the random number generator; 
                        If None, the random number generator is the RandomState instance used by np.random.""")

	parser.add_argument('--max_leaf_nodes', action='store', default='None', dest='max_leaf_nodes',
	                    help="""Grow a tree with max_leaf_nodes in best-first fashion. 
                        Best nodes are defined as relative reduction in impurity.
                         If None then unlimited number of leaf nodes.""")

	parser.add_argument('--min_impurity_decrease',  action='store', default="0.", dest='min_impurity_decrease',
	                    help="""A node will be split if this split induces a decrease of the impurity greater than or equal to this value.""")

	parser.add_argument('--class_weight', action='store', default='None', dest='class_weight',
	                    help="""Weights associated with classes in the form {class_label: weight}. 
                        If not given, all classes are supposed to have weight one. For multi-output problems, 
                        a list of dicts can be provided in the same order as the columns of y.""")

	parser.add_argument('--presort', action='store', default="False", dest='presort',
	                    help="""Whether to presort the data to speed up the finding of best splits in fitting. 
                        For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. 
                        When using either a smaller dataset or a restricted depth, this may speed up the training.""")

	args = parser.parse_args()
	main(args)



