"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Oct 06, 2019
Updated by: Omer Liberman

random_forest_regressor.py
==============================================================================
"""
import argparse
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

from cnvrg_sklearn_helper import train_with_cross_validation, train_without_cross_validation

def _cast_types(args):
    # x_val.
    if args.x_val != 'None':
        args.x_val = int(args.x_val)
    else:
        args.x_val = None

    # test_size
    args.test_size = float(args.test_size)

    # n_estimators.
    args.n_estimators = int(args.n_estimators)

    # max_depth.
    if args.max_depth == "None" or args.max_depth == 'None':
        args.max_depth = None
    else:
        args.max_depth = int(args.max_depth)

    # min_samples_split.
    try:
        args.min_samples_split = int(args.min_samples_split)
    except ValueError:
        args.min_samples_split = float(args.min_samples_split)

    # min_samples_leaf.
    try:
        args.min_samples_leaf = int(args.min_samples_leaf)
    except ValueError:
        args.min_samples_leaf = float(args.min_samples_leaf)

    # min_weight_fraction_leaf.
    args.min_weight_fraction_leaf = float(args.min_weight_fraction_leaf)

    # max_features.
    if args.max_features in ["auto", "sqrt", "log2"]:
        pass
    elif args.max_features == "None" or args.max_features == 'None':
        args.max_features = None
    else:
        try:
            args.max_features = float(args.max_features)
        except ValueError:
            args.max_features = int(args.max_features)

    # max_leaf_nodes.
    if args.max_leaf_nodes == "None" or args.max_leaf_nodes == 'None':
        args.max_leaf_nodes = None
    else:
        args.max_leaf_nodes = int(args.max_leaf_nodes)

    # min_impurity_decrease.
    args.min_impurity_decrease = float(args.min_impurity_decrease)

    # min_impurity_split.
    # args.min_impurity_split = float(args.min_impurity_split)

    # bootstrap.
    args.bootstrap = (args.bootstrap == "True" or args.bootstrap == 'True')

    # oob_score.
    args.oob_score = (args.oob_score == "True" or args.oob_score == 'True')

    # n_jobs.
  
    args.n_jobs = int(args.n_jobs)

    # random_state.
    if args.random_state == "None" or args.random_state == 'None':
        args.random_state = None
    else:
        args.random_state = int(args.random_state)

    # verbose.
    args.verbose = int(args.verbose)

    # warm_start.
    args.warm_start = (args.warm_start == "True" or args.warm_start == 'True')

    #  --- ---------------------------------------- --- #
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
    model = RandomForestRegressor(
                                    n_estimators=args.n_estimators,
                                    criterion=args.criterion,
                                    max_depth=args.max_depth,
                                    min_samples_split=args.min_samples_split,
                                    min_samples_leaf=args.min_samples_leaf,
                                    min_weight_fraction_leaf=args.min_weight_fraction_leaf,
                                    max_features=args.max_features,
                                    max_leaf_nodes=args.max_leaf_nodes,
                                    min_impurity_decrease=args.min_impurity_decrease,
                                    # min_impurity_split=args.min_impurity_split,
                                    bootstrap=args.bootstrap,
                                    oob_score=args.oob_score,
                                    n_jobs=args.n_jobs,
                                    random_state=args.random_state,
                                    verbose=args.verbose,
                                    warm_start=args.warm_start,
                                    )

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
    parser = argparse.ArgumentParser(description="""Random Forests Regressor""")
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
                        help="""String. The name of the output file which is a trained random forests model. Default is RandomForestModel.sav""")
    # ----- model's params.
    parser.add_argument('--n_estimators', action='store', default="10", dest='n_estimators',
                        help="""int: The number of trees in the forest. Default is 10""")

    parser.add_argument('--criterion', action='store', default='mse', dest='criterion',
                        help="""string: The function to measure the quality of a split.""")

    parser.add_argument('--max_depth', action='store', default="None", dest='max_depth',
                        help="""int: The maximum depth of the tree. If None, then nodes are expanded until all leaves 
                        are pure or until all leaves contain less than min_samples_split samples. Default is None""")

    # Might be int or float.
    parser.add_argument('--min_samples_split', action='store', default="2", dest='min_samples_split',
                        help="""int, float: The minimum number of samples required to split an internal node:
                        If int, then consider min_samples_split as the minimum number.
                        If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the 
                        minimum number of samples for each split.. Default is 2""")

    # Might be int or float.
    parser.add_argument('--min_samples_leaf', action='store', default="1", dest='min_samples_leaf',
                        help="""int, float: The minimum number of samples required to be at a leaf node. A split point 
                        at any depth will only be considered if it leaves at least min_samples_leaf training samples in
                        each of the left and right branches. This may have the effect of smoothing the model, especially 
                        in regression.
                        If int, then consider min_samples_leaf as the minimum number.
                        If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the 
                        minimum number of samples for each node. Default is 1""")

    parser.add_argument('--min_weight_fraction_leaf', action='store', default="0.",
                        dest='min_weight_fraction_leaf', help="""float: The minimum weighted fraction of the sum total 
                        of weights (of all the input samples) required to be at a leaf node. Samples have equal weight 
                        when sample_weight is not provided. Default is 0.""")

    # Might be int, float, string or None.
    parser.add_argument('--max_features', action='store', default="auto", dest='max_features',
                        help="""int, float, string, None: The number of features to consider when looking for the best 
                        split. 
                        If int, then consider max_features features at each split.
                        If float, then max_features is a fraction and int(max_features * n_features) features are 
                        considered at each split.
                        If ___auto___, then max_features=sqrt(n_features).
                        If ___sqrt___, then max_features=sqrt(n_features) (same as ___auto___).
                        If ___log2___, then max_features=log2(n_features).
                        If None, then max_features=n_features""")

    parser.add_argument('--max_leaf_nodes', action='store', default="None", dest='max_leaf_nodes',
                        help="""int, None,.Grow trees with max_leaf_nodes in best-first fashion Default is None.""")

    parser.add_argument('--min_impurity_decrease', action='store', default="0.", dest='min_impurity_decrease',
                        help="""float,.A node will be split if this split induces a decrease of the impurity greater
                         than or equal to this value. Default is 0..""")

    # parser.add_argument('--min_impurity_split', action='store', default="1e-7", dest='min_impurity_split',
    #                     help="""Deprecated since version 0.19: min_impurity_split has been deprecated in favor of
    #                     min_impurity_decrease in 0.19..""")

    parser.add_argument('--bootstrap', action='store', default="True", dest='bootstrap',
                        help="""Boolean. Whether bootstrap samples are used when building trees. If False, the whole 
                        dataset is used to build each tree. Default is True.""")

    parser.add_argument('--oob_score', action='store', default="False", dest='oob_score',
                        help="""Boolean. Whether to use out-of-bag samples to estimate the generalization accuracy.. 
                        Default is False.""")

    parser.add_argument('--n_jobs', action='store', default="1", dest='n_jobs',
                        help="""Integer. The number of jobs to run in parallel for both fit and predict. None means 1. 
                        Default is None.""")

    # Might be int, RandomState instance or None.
    parser.add_argument('--random_state', action='store', default="None", dest='random_state',
                        help="""int, RandomState instance or None. If int, random_state is the seed used by the random 
                        number generator; If RandomState instance, random_state is the random number generator; If None,
                         the random number generator is the RandomState instance used by np.random. Default is None.""")

    parser.add_argument('--verbose', action='store', default="0", dest='verbose',
                        help="""Integer. Controls the verbosity when fitting and predicting. Default is 0.""")

    parser.add_argument('--warm_start', action='store', default="True", dest='warm_start',
                        help="""Boolean. When set to True, reuse the solution of the previous call to fit and add more 
                        estimators to the ensemble, otherwise, just fit a whole new forest.. Default is True.""")


    args = parser.parse_args()

    main(args)