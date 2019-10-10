"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

cnvrg_random_forest_regressor_helper.py
-----------------------
This file performs training with or without cross-validation over
sklearn-models.
==============================================================================
"""
import pickle

from cnvrg import Experiment
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error

import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def train_with_cross_validation(model, train_set, test_set, folds, project_dir, output_model_name):
	"""
	This method enables sklearn algorithms to perform KFold-cross-validation.
	The method also initates the cnvrg.io experiment with all its metrics.
	:param model: SKlearn model object (initiated).
	:param train_set: tuple. (X_train, y_train). This is going to be used as a training set.
	:param test_set: tuple. (X_test, y_test). This is going to be used as a test set.
	:param folds: number of splits in the cross validation.
	:param project_dir: the path to the directory which indicates where to save the model.
	:param output_model_name: the name of the output model saved on the disk.
	:return: nothing.
	"""
	train_acc, train_loss = [], []
	kf = KFold(n_splits=folds)
	X, y = train_set

	# --- Training.
	for train_index, val_index in kf.split(X):
		X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
		y_train, y_val = y.iloc[train_index], y.iloc[val_index]
		model.fit(X_train, y_train)
		model.n_estimators += 1

		y_hat = model.predict(X_val)  # y_hat is a.k.a y_pred

		acc = accuracy_score(y_val, y_hat)
		loss = mean_squared_error(y_val, y_hat)

		train_acc.append(acc)
		train_loss.append(loss)

	# --- Testing.
	X_test, y_test = test_set
	y_pred = model.predict(X_test)
	test_acc = accuracy_score(y_test, y_pred)
	test_loss = mean_squared_error(y_test, y_pred)

	exp = Experiment()
	exp.log_param("model", output_model_name)
	exp.log_param("folds", folds)
	exp.log_metric("train_acc", train_acc)
	exp.log_metric("train_loss", train_loss)
	exp.log_param("test_acc", test_acc)
	exp.log_param("test_loss", test_loss)

	# Save model.
	output_file_name = project_dir + "/" + output_model_name if project_dir is not None else output_model_name
	pickle.dump(model, open(output_file_name, 'wb'))


def train_without_cross_validation(model, train_set, test_set, project_dir, output_model_name):
	"""
	The method also initates the cnvrg.io experiment with all its metrics.
	:param model: SKlearn model object (initiated).
	:param train_set: tuple. (X_train, y_train). This is going to be used as a training set.
	:param test_set: tuple. (X_test, y_test). This is going to be used as a test set.
	:param project_dir: the path to the directory which indicates where to save the model.
	:param output_model_name: the name of the output model saved on the disk.
	:return: nothing.
	"""
	X_train, y_train = train_set

	# --- Training.
	model.fit(X_train, y_train)

	y_hat = model.predict(X_train)  # y_hat is a.k.a y_pred

	train_acc = accuracy_score(y_train, y_hat)
	train_loss = mean_squared_error(y_train, y_hat)

	# --- Testing.
	X_test, y_test = test_set
	y_pred = model.predict(X_test)
	test_acc = accuracy_score(y_test, y_pred)
	test_loss = mean_squared_error(y_test, y_pred)

	exp = Experiment()
	exp.log_param("model", output_model_name)
	exp.log_param("train_acc", train_acc)
	exp.log_param("train_loss", train_loss)
	exp.log_param("test_acc", test_acc)
	exp.log_param("test_loss", test_loss)

	# Save model.
	output_file_name = project_dir + "/" + output_model_name if project_dir is not None else output_model_name
	pickle.dump(model, open(output_file_name, 'wb'))
