"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

utils/sk_trainer.py
==============================================================================
"""
import os
import pickle

import numpy as np
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar
from cnvrg.charts.pandas_analyzer import MatrixHeatmap

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, \
	accuracy_score, zero_one_loss, f1_score, mean_absolute_error, mean_squared_error, \
	r2_score, log_loss

"""
SKTrainerClassification
-----------------------
This module is used for training sk-learn module.

It runs the entire training, validation and testing phases.
At the end, it logs metrics and parameters to cnvrg experiment, and 
save a trained model.
"""
class SKTrainerClassification:
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
		except Exception:
			self.__cnvrg_env = False

	def run(self):
		"""
		This method runs the whole process:
		training, validation, testing and model's saving.

		:return: none.
		"""
		if self.__train_with_cross_validation:
			self.__metrics['cross_validation_folds'] = (int, self.__cross_validation_folds)
			self.__train_with_cv()

		else:
			self.__train_without_cv()

		self.__save_model()

	def __plot(self, predicted_y):
		"""
		This method controls the visualization and metrics plots.
		:param predicted_y: the model's prediction for the test set.

		:return: none
		"""
		self.__plot_correlation_matrix()
		self.__plot_feature_importance()
		self.__plot_classification_report(y_test_pred=predicted_y)
		self.__plot_confusion_matrix(y_test_pred=predicted_y)
		self.__plot_ROC_curve(y_test_pred=predicted_y)
		self.__log_accuracies_and_losses()

	def __train_with_cv(self):
		"""
		This method enables sk-learn algorithms to perform KFold-cross-validation.
		The method also initiates the cnvrg experiment with all its metrics.

		:return: none.
		"""
		def get_train_loss_type(shortcut_name):
			"""
			Gets the title of the loss type and return the name of the
			loss type matched the ''cross_validate'' function.
			"""
			if shortcut_name == 'F1':
				return 'f1'
			elif shortcut_name == 'LOG':
				return 'neg_log_loss'
			elif shortcut_name == 'MAE':
				return 'neg_mean_absolute_error'
			elif shortcut_name == 'MSE':
				return 'neg_mean_squared_error'
			elif shortcut_name == 'RMSE':
				return 'neg_root_mean_squared_error'
			elif shortcut_name == 'R2':
				return 'r2'

		# Training.
		train_loss_type = get_train_loss_type(self.__metrics['train_loss_type'][1]) # (_type, _value)
		scores = cross_validate(
								estimator=self.__model,
								X=self.__X_train.values,
								y=self.__Y_train.values,
								cv=self.__cross_validation_folds,
								return_train_score=True,
								scoring=[train_loss_type, 'accuracy'],
								return_estimator=True)
		train_acc_cv = scores['train_accuracy']
		val_acc_cv = scores['test_accuracy']
		train_err_cv = scores['train_' + train_loss_type]
		val_err_cv = scores['test_' + train_loss_type]
		if train_loss_type.startswith('neg'):
			train_err_cv *= (-1)
			val_err_cv *= (-1)
		self.__model = scores['estimator'][-1]  # the trained model.

		# Testing
		y_pred_test = self.__model.predict(self.__X_test.values)
		test_acc = accuracy_score(self.__Y_test.values, y_pred_test)
		test_loss_func = SKTrainerClassification.get_loss_function(self.__metrics['test_loss_type'][1]) # (_type, _value)
		test_loss = test_loss_func(self.__Y_test.values, y_pred_test)

		# Updating metrics values.
		self.__metrics.update({
			'train_acc': (np.ndarray, train_acc_cv),
			'train_loss': (np.ndarray, train_err_cv),
			'validation_acc': (np.ndarray, val_acc_cv),
			'validation_loss': (np.ndarray, val_err_cv),
			'test_acc': (float, test_acc),
			'test_loss': (float, test_loss),
		})

		# Plot metrics to cnvrg environment.
		self.__plot(y_pred_test)

	def __train_without_cv(self):
		"""
		This method enables sk-learn algorithms to performs training & testing.
		The method also initiates the cnvrg experiment with all its metrics.
		:return: none.
		"""
		# Training.
		self.__model.fit(self.__X_train.values, self.__Y_train.values)
		y_pred_train = self.__model.predict(self.__X_train.values)
		train_acc = accuracy_score(self.__Y_train, y_pred_train)
		train_loss_func = SKTrainerClassification.get_loss_function(self.__metrics['train_loss_type'][1]) # (_type, _value)
		train_loss = train_loss_func(self.__Y_train, y_pred_train)

		# Testing
		y_pred_test = self.__model.predict(self.__X_test.values)
		test_acc = accuracy_score(self.__Y_test, y_pred_test)
		test_loss_func = SKTrainerClassification.get_loss_function(self.__metrics['test_loss_type'][1]) # (_type, _value)
		test_loss = test_loss_func(self.__Y_test, y_pred_test)

		# Updating metrics values.
		self.__metrics.update({
			'train_acc': (float, train_acc),
			'train_loss': (float, train_loss),
			'test_acc': (float, test_acc),
			'test_loss': (float, test_loss)})

		# Plot metrics to cnvrg environment.
		self.__plot(y_pred_test)

	@staticmethod
	def get_loss_function(shortcut_name):
		"""
		returns the loss function for the test phase.
		"""
		if shortcut_name == 'F1':
			return f1_score
		elif shortcut_name == 'LOG':
			return log_loss
		elif shortcut_name == 'MAE':
			return mean_absolute_error
		elif shortcut_name == 'MSE':
			return mean_squared_error
		elif shortcut_name == 'R2':
			return r2_score
		elif shortcut_name == 'zero_one_loss':
			return zero_one_loss
		else:
			raise Exception('Undefined error type.')

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

	def __plot_classification_report(self, y_test_pred):
		"""
		This method creates the classification report.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the table to stdin.

		:param y_test_pred: the predicted values for the test set.
		:return: none.
		"""
		classification_report_test_set = classification_report(self.__Y_test, y_test_pred, output_dict=True)  # dict
		if self.__cnvrg_env:
			classification_report_as_array = self.__helper_plot_classification_report(classification_report_test_set)
			self.__experiment.log_chart("Test Set - Classification Report", data=Heatmap(z=classification_report_as_array),
										y_ticks=self.__labels,
										x_ticks=["precision", "recall", "f1-score", "support"])
		else:
			print("---Classification Report---")
			print(classification_report_test_set)
			print("---------------------------")


	def __plot_confusion_matrix(self, y_test_pred=None):
		"""
		This method creates the confusion matrix.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the values to stdin.

		:param y_test_pred: the predicted values for the test set.
		:return: none.
		"""
		if self.__Y_test is not None and y_test_pred is not None:
			confusion_matrix_test_set = confusion_matrix(self.__Y_test, y_test_pred)  # array
			confusion_matrix_test_set = self.__helper_plot_confusion_matrix(confusion_matrix_test_set)

			if self.__cnvrg_env:
				self.__experiment.log_chart("Test Set - confusion matrix", data=Heatmap(z=confusion_matrix_test_set))
			else:
				print('---Confusion Matrix---')
				print(confusion_matrix_test_set)
				print('----------------------')

	def __plot_ROC_curve(self, y_test_pred):
		"""
		This method calculates the ROC curve.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the values to stdin.

		Note: ROC curve can be produced if and only if there exactly 2 labels in the
		test set. Otherwise, the method ends.

		:param y_test_pred: the predicted values for the test set.
		:return: none.
		"""
		num_of_labels_in_test_set = len(set(self.__Y_test))
		if num_of_labels_in_test_set != 2:  ## roc curve works only for 2 labels.
			return

		fpr, tpr, _ = roc_curve(self.__Y_test, y_test_pred)

		if self.__cnvrg_env:
			self.__experiment.log_metric(key='ROC curve', Ys=tpr.tolist(), Xs=fpr.tolist())
		else:
			print("----ROC----")
			print("FPRs: {fpr}".format(fpr=fpr))
			print("TPRs: {tpr}".format(tpr=tpr))
			print("-----------")

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

	def __log_accuracies_and_losses(self):
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

	def __helper_plot_confusion_matrix(self, conf_matrix):
		"""
		This method is an helper for '''self.__plot_confusion_matrix()'''.
		It gets the conf_matrix of sk-learn and prepares it for structure required
		for cnvrg Heatmap feature (triplets of x,y,z).

		:param conf_matrix: the output of '''sklearn.metrics.confusion_matrix'''.
		:return: an array of triplets (x,y,z) where x,y are the coordinates in the table
		and z is the value.
		"""
		digits_to_round = self.__metrics['digits_to_round'][1] # (_type, _value)
		output = []
		for y in range(len(conf_matrix)):
			for x in range(len(conf_matrix[y])):
				z = round(float(conf_matrix[x][y]), digits_to_round)
				output.append((x, y, z))
		return output

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

			self.__metrics[metric] = (_type, _value) # updating the previous value.


	def __helper_plot_classification_report(self, classification_report_dict):
		"""
		This method is an helper for '''__plot_classification_report'''.
		It gets a dictionary which is the output of '''sklearn.metrics.confusion_matrix'''.
		It convert it to list of triplets which fits the requirements of cnvrg's Heatmap feature.

		:param classification_report_dict: a dictionary.
		:return: list of triplets (x,y,z) where x,y indicates a coordinates and z is the value.
		"""
		digits_to_round = self.__metrics['digits_to_round'][1] # (_type, _value)
		output = []

		for k, v in classification_report_dict.items():
			if k in self.__labels:
				output.append(list(v.values()))

		values = []
		for y in range(len(output)):
			for x in range(len(output[y])):
				z = round(output[y][x], digits_to_round)
				values.append((x, y, z))
		return values



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
		except Exception:
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
