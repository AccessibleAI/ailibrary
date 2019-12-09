"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

regression_helper.py
-----------------------
This file performs training with or without cross-validation over SK-learn models.
==============================================================================
"""
import os
import pickle
import numpy as np

from cnvrg import Experiment
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def train_with_cross_validation(model, train_set, test_set, folds, project_dir, output_model_name, testing_mode):
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
	train_loss_MSE, train_loss_MAE, train_loss_R2 = [], [], []
	kf = KFold(n_splits=folds)
	X, y = train_set

	# --- Training.
	for train_index, val_index in kf.split(X):
		X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
		y_train, y_val = y.iloc[train_index], y.iloc[val_index]

		model = model.fit(X_train, y_train)
		y_hat = model.predict(X_val)  # y_hat is a.k.a y_pred

		mse = mean_squared_error(y_val, y_hat)
		mae = mean_absolute_error(y_val, y_hat)
		r2 = r2_score(y_val, y_hat)

		train_loss_MSE.append(mse)
		train_loss_MAE.append(mae)
		train_loss_R2.append(r2)

	# --- Testing.
	X_test, y_test = test_set
	y_pred = model.predict(X_test)
	test_loss_MSE = mean_squared_error(y_test, y_pred)
	test_loss_MAE = mean_absolute_error(y_test, y_pred)
	test_loss_R2 = r2_score(y_test, y_pred)

	if not testing_mode:
		exp = Experiment()
		exp.log_param("model", output_model_name)
		exp.log_param("folds", folds)

		exp.log_metric("train_loss_MSE", train_loss_MSE)
		exp.log_metric("train_loss_MAE", train_loss_MAE)
		exp.log_metric("train_loss_R2", train_loss_R2)

		exp.log_metric("test_loss_MSE", test_loss_MSE)
		exp.log_metric("test_loss_MAE", test_loss_MAE)
		exp.log_metric("test_loss_R2", test_loss_R2)
	else:
		print("Model: {model}\n"
			  "Folds: {folds}\n"
			  "train_loss_MSE={train_loss_MSE}\n"
			  "train_loss_MAE={train_loss_MAE}\n"
			  "train_loss_R2={train_loss_R2}\n"
			  "test_loss_MSE={test_loss_MSE}\n"
			  "test_loss_MAE={test_loss_MAE}\n"
			  "test_loss_R2={test_loss_R2}".format(
			model=output_model_name, folds=folds, train_loss_MSE=train_loss_MSE, train_loss_MAE=train_loss_MAE,
		train_loss_R2=train_loss_R2, test_loss_MSE=test_loss_MSE, test_loss_MAE=test_loss_MAE, test_loss_R2=test_loss_R2))

	# Save model.
	output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") is not None else output_model_name
	pickle.dump(model, open(output_file_name, 'wb'))

	if not testing_mode:
		os.system("ls -la {}".format(os.environ.get("CNVRG_PROJECT_PATH")))


def train_without_cross_validation(model, train_set, test_set, project_dir, output_model_name, testing_mode):
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
	model = model.fit(X_train, y_train)
	y_hat = model.predict(X_train)  # y_hat is a.k.a y_pred

	train_loss_MSE = mean_squared_error(y_train, y_hat)
	train_loss_MAE = mean_absolute_error(y_train, y_hat)
	train_loss_R2 = r2_score(y_train, y_hat)

	# --- Testing.
	X_test, y_test = test_set
	y_pred = model.predict(X_test)
	test_loss_MSE = mean_squared_error(y_test, y_pred)
	test_loss_MAE = mean_absolute_error(y_test, y_pred)
	test_loss_R2 = r2_score(y_test, y_pred)

	if not testing_mode:
		exp = Experiment()
		exp.log_param("model", output_model_name)

		exp.log_metric("train_loss_MSE", train_loss_MSE)
		exp.log_metric("train_loss_MAE", train_loss_MAE)
		exp.log_metric("train_loss_R2", train_loss_R2)

		exp.log_metric("test_loss_MSE", test_loss_MSE)
		exp.log_metric("test_loss_MAE", test_loss_MAE)
		exp.log_metric("test_loss_R2", test_loss_R2)
	else:
		print("Model: {model}\n"
			  "train_loss_MSE={train_loss_MSE}\n"
			  "train_loss_MAE={train_loss_MAE}\n"
			  "train_loss_R2={train_loss_R2}\n"
			  "test_loss_MSE={test_loss_MSE}\n"
			  "test_loss_MAE={test_loss_MAE}\n"
			  "test_loss_R2={test_loss_R2}".format(
			model=output_model_name, train_loss_MSE=train_loss_MSE, train_loss_MAE=train_loss_MAE,
		train_loss_R2=train_loss_R2, test_loss_MSE=test_loss_MSE, test_loss_MAE=test_loss_MAE, test_loss_R2=test_loss_R2))

	# Save model.
	output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") is not None else output_model_name
	pickle.dump(model, open(output_file_name, 'wb'))

	if not testing_mode:
		os.system("ls -la {}".format(os.environ.get("CNVRG_PROJECT_PATH")))
