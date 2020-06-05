"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

sk_trainer_regression.py
==============================================================================
"""
import os
import pickle

import numpy as np
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar
from cnvrg.charts.pandas_analyzer import MatrixHeatmap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, \
	accuracy_score, zero_one_loss, f1_score, mean_absolute_error, mean_squared_error, \
	r2_score, log_loss


"""
SKTrainerRegression
-----------------------
This module is used for training sk-learn module.

Note - cross validation is not allowed in regression algorithms.

It runs the entire training, validation and testing phases.
At the end, it logs metrics and parameters to cnvrg experiment, and 
save a trained model.
"""
class SKTrainerRegression:
	REGRESSION_TYPE = ['linear', 'logistic']

	def __init__(self,
				 sk_learn_model_object,
				 path_to_csv_file,
				 test_size,
				 output_model_name,
				 train_loss_type,
				 test_loss_type,
				 digits_to_round,
				 folds=None):
		"""
		Inits the classification trainer object.
		:param sk_learn_model_object: an initiated sklearn model.
		:param path_to_csv_file: (str) path to the data set, a csv file.
		:param test_size: (float) (default: 0.2) the size of the test set.
		:param output_model_name (str) the output model name file.
		:param train_loss_type (str) the name of the loss function for train set.
		:param test_loss_type (str) the name of the loss function for the test set.
		:param digits_to_round (int) the number of digits to round the output metrics.
		:param folds: (int) (default: None) indicates whether to perform cross validation or not.
		"""
		self.__model = sk_learn_model_object

		data = pd.read_csv(path_to_csv_file, index_col=0)
		X = data.iloc[:, :-1]
		y = data.iloc[:, -1]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
		self.__X_train, self.__Y_train = X_train, y_train
		self.__X_test, self.__Y_test = X_test, y_test
		self.__all_data_concatenated = data

		self.__cross_validation_folds = folds
		self.__train_with_cross_validation = (folds is not None)

		self.__features = list(self.__X_train.columns)
		self.__labels = [str(label) for label in list(y)]

		train_set_size = (int, len(self.__Y_train))
		test_set_size = (int, len(self.__Y_test))
		output_model_file = (str, output_model_name)
		train_loss_type = (str, train_loss_type)
		test_loss_type = (str, test_loss_type)
		digits_to_round = (int, digits_to_round)

		self.__metrics = {
			'train_set_size': train_set_size,
			'test_set_size': test_set_size,
			'output_model_file': output_model_file,
			'train_loss_type': train_loss_type,
			'test_loss_type': test_loss_type,
			'digits_to_round': digits_to_round
		}

		try:
			self.__experiment = Experiment()
			self.__cnvrg_env = True
		except:
			self.__cnvrg_env = False

		#self.__regression_type = SKTrainerRegression.REGRESSION_TYPE[regression_type]
		self.__coef, self.__intercept = None, None

	def run(self):
		"""
		This method runs the whole process:
		training, testing and model's saving.

		When a validation is requested it prints a message
		which states validation is not allowed in regression algorithms,
		and runs regular training.

		:return: none.
		"""
		try:
			self.__coef = self.__model.coef_
		except AttributeError:
			pass

		try:
			self.__intercept = self.__model.intercept_
		except AttributeError:
			pass

		if self.__train_with_cross_validation:
			print("Error: Cross Validation is not allowed in regression algorithms.")
			print("Runs regular training...")

		self.__train_without_cv()
		self.__save_model()

	def __plot(self, predicted_y):
		"""
		This method controls the visualization and metrics plots.
		:param predicted_y: the model's prediction for the test set.

		:return: none
		"""
		self.__plot_feature_importance()
		self.__plot_correlation_matrix()
		# self.__plot_feature_vs_feature()
		self.__log_accuracies_and_errors()

	def __train_without_cv(self):
		"""
		This method enables sk-learn algorithms to performs training & testing.
		The method also initiates the cnvrg experiment with all its metrics.
		:return: none.
		"""
		# Training.
		self.__model.fit(self.__X_train.values, self.__Y_train.values)
		y_pred_train = self.__model.predict(self.__X_train.values)
		train_loss_MSE = mean_squared_error(self.__Y_train, y_pred_train)
		train_loss_MAE = mean_absolute_error(self.__Y_train, y_pred_train)
		train_loss_R2 = r2_score(self.__Y_train, y_pred_train)

		# Testing.
		y_pred_test = self.__model.predict(self.__X_test)
		test_loss_MSE = mean_squared_error(self.__Y_test, y_pred_test)
		test_loss_MAE = mean_absolute_error(self.__Y_test, y_pred_test)
		test_loss_R2 = r2_score(self.__Y_test, y_pred_test)

		# Updating metrics values.
		self.__metrics.update({
			'train_loss_mae': (float, train_loss_MAE),
			'train_loss_mse': (float, train_loss_MSE),
			'train_loss_r2': (float, train_loss_R2),
			'test_loss_mse': (float, test_loss_MSE),
			'test_loss_mae': (float, test_loss_MAE),
			'test_loss_r2': (float, test_loss_R2)})

		# Plot metrics to cnvrg environment.
		self.__plot(y_pred_test)


	def __plot_feature_importance(self):
		"""
		This method creates the feature importance table.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the table to stdin.

		Note: not all models has the option to produce the feature importance table.
		If the model can't, an exception is thrown.

		:return: none.
		"""
		try:
			feature_importance_values = getattr(self.__model, "feature_importances_")
			if self.__cnvrg_env:
				self.__experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance',
											data=Bar(x=self.__features, y=feature_importance_values))
			else:
				print("---Feature Importance---")
				print(feature_importance_values)
				print("------------------------")
		except AttributeError:
			pass


	def __log_accuracies_and_errors(self):
		"""
		This method logs the accuracies and the losses.
		If a cnvrg environment is recognized, it produces metrics and params at cnvrg experiment.
		Otherwise it prints the values to stdin.

		Note: it rounds the values in the table to 2 decimal digits.
		Otherwise, the UI seems broken.

		:return: none.
		"""
		self.__log_accuracies_and_losses_helper_rounding()

		for metric in self.__metrics:
			# Uninitialized metric.
			if self.__metrics[metric] is None:
				continue

			_key = metric
			_type, _value = self.__metrics[metric]

			# Metrics.
			if _type in [list, np.ndarray]:
				if self.__cnvrg_env:
					self.__experiment.log_metric(_key, _value, grouping=[_key] * len(_value))
				else:
					print("Metric: {key} : {value}".format(key=_key, value=_value))
			# Params.
			else:
				if self.__cnvrg_env:
					self.__experiment.log_param(_key, _value)
				else:
					print("Param: {key} : {value}".format(key=_key, value=_value))

	def __log_accuracies_and_losses_helper_rounding(self):
		"""
		This method is an helper for '''self.__log_accuracies_and_losses()'''.
		It rounds the values such that they'll be logged easily to the charts.

		:return: none.
		"""
		digits_to_round = self.__metrics['digits_to_round'][1] # (_type, _value)

		for metric in self.__metrics.keys():
			_key = metric
			_type, _value = self.__metrics[metric]

			# Lists & Arrays.
			if _type in [list, np.ndarray]:
				if _type is np.ndarray:
					_value = _value.tolist()

				for ind in range(len(_value)):
					_value[ind] = round(_value[ind], digits_to_round)

			# int & floats.
			elif _type in [int, float]:
				_value = round(_value, digits_to_round)

			self.__metrics[metric] = (_type, _value)  # updating the previous value.

	def __save_model(self):
		"""
		This method saves the trained model object.
		If a cnvrg environment is recognized, it saves the model in the project directory.
		Otherwise, it saves the model in the path given at '''self.__metrics['output_model_file']'''.

		:return: none.
		"""
		output_model_name = self.__metrics['output_model_file'][1] # (_type, _value)
		output_file_name = os.environ.get("CNVRG_WORKDIR") + "/" + output_model_name \
			if os.environ.get("CNVRG_WORKDIR") is not None else output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))

	"""training & testing methods"""

	def __plot_correlation_matrix(self):
		"""
		This method calculates the correlation matrix.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the values to stdin.

		Note: it rounds the values in the table to 2 decimal digits.
		Otherwise, the UI seems broken.

		:return: none.
		"""
		data = self.__all_data_concatenated
		correlation = data.corr()

		if self.__cnvrg_env:
			self.__experiment.log_chart("correlation",
										# must round to 2 numbers otherwise it gets out of the table:
										[MatrixHeatmap(np.round(correlation.values, 2))],
										x_ticks=correlation.index.tolist(),
										y_ticks=correlation.index.tolist())
		else:
			print("--- Correlation ---")
			print(correlation)
			print("-------------------")
