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

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression

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

    # penalty

    # dual
    args.dual = (args.dual == "True" or args.dual == 'True')

    # tol
    args.tol = float(args.tol)

    # C
    args.C = float(args.C)

    # fit_intercept
    args.fit_intercept = (args.fit_intercept == "True" or args.fit_intercept == 'True')

    # intercept_scaling
    args.intercept_scaling = float(args.intercept_scaling)

    # might be dict - for now only None
    # class_weight
    if args.class_weight == "None" or args.class_weight == 'None':
        args.class_weight = None

    # random_state
    if args.random_state == "None" or args.random_state == 'None':
        args.random_state = None
    else:
        args.random_state = int(args.random_state)

    # solver

    # max_iter
    args.max_iter = int(args.max_iter)

    # multi_class

    # verbose
    args.verbose = int(args.verbose)

    # warm_start
    args.warm_start = (args.warm_start == "True" or args.warm_start == 'True')

    # n_jobs
    if args.n_jobs == "None" or args.n_jobs == 'None':
        args.n_jobs = None
    else:
        args.n_jobs = int(args.n_jobs)

    # l1_ratio
    # if args.l1_ratio == "None" or args.l1_ratio == 'None':
    #     args.l1_ratio = None
    # else:
    #     args.l1_ratio = float(args.l1_ratio)

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
    model = LogisticRegression(
                                    penalty=args.penalty,
                                    dual=args.dual,
                                    tol=args.tol,
                                    C=args.C,
                                    fit_intercept=args.fit_intercept,
                                    intercept_scaling=args.intercept_scaling,
                                    class_weight=args.class_weight,
                                    random_state=args.random_state,
                                    solver=args.solver,
                                    max_iter=args.max_iter,
                                    multi_class=args.multi_class,
                                    warm_start=args.warm_start,
                                    n_jobs=args.n_jobs,
                                    verbose=args.verbose
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
    parser = argparse.ArgumentParser(description="""logisticregression""")
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

    parser.add_argument('--output_model', action='store', default="logistic_regression_model.sav", dest='output_model',
                        help="""String. The name of the output file which is a trained random forests model. Default is logistic_regression_model.sav""")

    # ----- model's params.
    parser.add_argument('--penalty', action='store', default='l2', dest='penalty',
                        help="""str, Used to specify the norm used in the penalization. The ‘newton-cg’, 
                        ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver.
                         If ‘none’ (not supported by the liblinear solver), no regularization is applied.""")

    parser.add_argument('--dual', action='store', default="False", dest='dual',
                        help="""bool, Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.""")

    parser.add_argument('--tol', action='store', default="1e-4", dest='tol',
                        help="""float, Tolerance for stopping criteria.""")

    parser.add_argument('--C', action='store', default="1.0", dest='C',
                        help="""float, Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. """)

    parser.add_argument('--fit_intercept', action='store', default="True",dest='fit_intercept',
                        help="""bool, Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.""")

    parser.add_argument('--intercept_scaling', action='store', default="1", dest='intercept_scaling',
                        help="""float, Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], 
                        i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. 
                        The intercept becomes intercept_scaling * synthetic_feature_weight.""")

    parser.add_argument('--class_weight', action='store', default="None", dest='class_weight',
                        help="""dict, Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
                        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).""")

    parser.add_argument('--random_state', action='store', default="None", dest='random_state',
                        help="""int,The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; 
                        If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. 
                        Used when solver == ‘sag’ or ‘liblinear’.""")

    parser.add_argument('--solver', action='store', default="liblinear", dest='solver',
                        help="""Boolean. Algorithm to use in the optimization problem.
                        - For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
                        - For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
                        - ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
                        - ‘liblinear’ and ‘saga’ also handle L1 penalty
                        - ‘saga’ also supports ‘elasticnet’ penalty
                        - ‘liblinear’ does not handle no penalty. """)

    parser.add_argument('--max_iter', action='store', default="100", dest='max_iter',
                        help="""int. Maximum number of iterations taken for the solvers to converge.""")

    parser.add_argument('--multi_class', action='store', default="ovr", dest='multi_class',
                        help="""If the option chosen is ‘ovr’, then a binary problem is fit for each label. 
                        For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. 
                        ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.""")

    parser.add_argument('--verbose', action='store', default="0", dest='verbose',
                        help="""For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.""")

    parser.add_argument('--warm_start', action='store', default="False", dest='warm_start',
                        help="""bool. When set to True, reuse the solution of sthe previous call to fit as initialization, otherwise, 
                        just erase the previous solution. Useless for liblinear solver.""")

    parser.add_argument('--n_jobs', action='store', default="None", dest='n_jobs',
                        help="""int.  Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. 
                        This parameter is ignored when the solver is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. 
                        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors""")

    # parser.add_argument('--l1_ratio', action='store', default="None", dest='l1_ratio',
    #                     help="""float, The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'`.
    #                     Setting ``l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'.
    #                     For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.""")

    args = parser.parse_args()

    main(args)