"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainer.py
==============================================================================
"""
import os
import pickle
import numpy as np

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar, Scatterplot
from cnvrg.charts.pandas_analyzer import PandasAnalyzer

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_squared_error, r2_score, mean_absolute_error


class SKTrainerRegression:
	DIGITS_TO_ROUND = 3

	def __init__(self, model, train_set, test_set, output_model_name, testing_mode, folds=None):
		self.__model = model
		self.__x_train, self.__y_train = train_set
		self.__train_set_size = len(self.__y_train)
		self.__x_test, self.__y_test = test_set
		self.__test_set_size = len(self.__y_test)
		self.__output_model_name = output_model_name
		self.__testing_mode = testing_mode
		self.__cross_val_folds = folds
		self.__is_cross_val = (folds is not None)
		self.__features = list(self.__x_train.columns)
		self.__labels = [str(l) for l in list(set(self.__y_train).union(set(self.__y_test)))]

		self.__model.fit(self.__x_train, self.__y_train)
		self.__coef = self.__model.coef_
		self.__intercept = self.__model.intercept_
		self.__y_pred = None

		self.__experiment = Experiment.init('test_charts')  # todo -> delete / replace with: self.__experiment = Experiment()

		self.__metrics = {'model': self.__output_model_name}
		if self.__is_cross_val:
			self.__metrics['folds'] = self.__cross_val_folds

	def run(self):
		""" runs the training & testing methods. """
		if self.__is_cross_val is True:
			self.__train_with_cross_validation()
		else:
			self.__train_without_cross_validation()
		self.__save_model()

	def __plot_all(self, y_test_pred):
		self.__plot_accuracies_and_errors()
		self.__plot_true_against_prediction()

	"""training & testing methods"""

	def __train_with_cross_validation(self):
		"""
		This method enables sk-learn algorithms to perform KFold-cross-validation.
		The method also initiates the cnvrg experiment with all its metrics.
		"""
		train_acc, train_loss = [], []
		kf = KFold(n_splits=self.__cross_val_folds)
		scores = cross_validate(estimator=self.__model,
								X=self.__x_train,
								y=self.__y_train,
								cv=self.__cross_val_folds,
								return_train_score=True,
								scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'accuracy'],
								return_estimator=True)

		train_err_cv_mse = (-1) * scores['train_neg_mean_squared_error']
		train_err_cv_mae = (-1) * scores['train_neg_mean_absolute_error']
		train_err_cv_r2 = scores['train_r2']

		val_acc_cv = scores['test_accuracy']
		val_err_cv_mse = (-1) * scores['test_neg_mean_squared_error']
		val_err_cv_mae = (-1) * scores['test_neg_mean_absolute_error']
		val_err_cv_r2 = scores['test_r2']

		self.__model = scores['estimator'][-1]
		self.__y_pred = self.__model.predict(self.__x_test)
		test_acc = accuracy_score(self.__y_test, self.__y_pred)
		test_loss = mean_squared_error(self.__y_test, self.__y_pred)
		self.__metrics.update({
			'train_loss_mae': train_err_cv_mae,
			'train_loss_mse': train_err_cv_mse,
			'train_loss_r2': train_err_cv_r2,
			'validation_acc': val_acc_cv,
			'val_loss_mae': val_err_cv_mae,
			'val_loss_mse': val_err_cv_mse,
			'val_loss_r2': val_err_cv_r2,
			'test_acc': test_acc,
			'test_loss_mse': test_loss
		})
		self.__plot_all(self.__y_pred)

	def __train_without_cross_validation(self):
		"""
		The method also initiates the cnvrg experiment with all its metrics.
		"""
		y_hat = self.__model.predict(self.__x_train)  # y_hat is a.k.a y_pred

		train_loss_MSE = mean_squared_error(self.__y_train, y_hat)
		train_loss_MAE = mean_absolute_error(self.__y_train, y_hat)
		train_loss_R2 = r2_score(self.__y_train, y_hat)

		self.__y_pred = self.__model.predict(self.__x_test)
		test_loss_MSE = mean_squared_error(self.__y_test, self.__y_pred)
		test_loss_MAE = mean_absolute_error(self.__y_test, self.__y_pred)
		test_loss_R2 = r2_score(self.__y_test, self.__y_pred)

		self.__metrics.update({
			'train_loss_mae': train_loss_MAE,
			'train_loss_mse': train_loss_MSE,
			'train_loss_r2': train_loss_R2,
			'test_loss_mse': test_loss_MSE,
			'test_loss_mae': test_loss_MAE,
			'test_loss_r2': test_loss_R2
		})
		self.__plot_all(self.__y_pred)

	def __plot_true_against_prediction(self):
		a, b = self.__coef[0], self.__intercept
		x = np.linspace(0, len(self.__x_test), 1000)
		y = a * x + b

		self.__experiment.log_metric(key="Regression Line", Xs=x.tolist(), Ys=y.tolist())

	def __plot_accuracies_and_errors(self):
		"""Plots the metrics."""
		if self.__testing_mode is True:
			print("Model: {model}\n"
				  "train_acc={train_acc}\n"
				  "train_loss={train_loss}\n"
				  "test_acc={test_acc}\n"
				  "test_loss={test_loss}".format(
				model=self.__metrics['model'], train_acc=self.__metrics['train_acc'], train_loss=self.__metrics['train_loss'],
				test_acc=self.__metrics['test_acc'], test_loss=self.__metrics['test_loss']))
			if self.__is_cross_val is True:
				print("Folds: {folds}\n".format(folds=self.__metrics['folds']))

		else: # testing mode is off.
			for k, v in self.__metrics.items():
				self.__plot_accuracies_and_errors_helper()
				if isinstance(v, list):
					self.__experiment.log_metric(k, v)
				else:
					self.__experiment.log_param(k, v)

	def __plot_accuracies_and_errors_helper(self):
		for k, v in self.__metrics.items():
			if isinstance(v, float):
				self.__metrics[k] = round(self.__metrics[k], SKTrainerRegression.DIGITS_TO_ROUND)

	def __save_model(self):
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + self.__output_model_name if os.environ.get("CNVRG_PROJECT_PATH") \
																									is not None else self.__output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))
