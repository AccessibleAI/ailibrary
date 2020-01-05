"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainer.py
==============================================================================
"""
import os
import pickle

import numpy
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar
from cnvrg.charts.pandas_analyzer import PandasAnalyzer

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_squared_error


class SKTrainer:
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
		# self.__importance = self.__model.feature_importances_

		self.__experiment = Experiment()

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
								scoring=['neg_mean_squared_error', 'accuracy'],
								return_estimator=True)

		train_acc_cv = scores['train_accuracy']
		train_err_cv = (-1) * scores['train_neg_mean_squared_error']
		val_acc_cv = scores['test_accuracy']
		val_err_cv = (-1) * scores['test_neg_mean_squared_error']
		self.__model = scores['estimator'][-1]

		y_pred = self.__model.predict(self.__x_test)
		test_acc = accuracy_score(self.__y_test, y_pred)
		test_loss = mean_squared_error(self.__y_test, y_pred)
		self.__metrics.update({
			'train_acc': train_acc_cv,
			'train_loss': train_err_cv,
			'validation_acc': val_acc_cv,
			'validation_loss': val_err_cv,
			'test_acc': test_acc,
			'test_loss': test_loss
		})
		self.__plot_all(y_pred)

	def __train_without_cross_validation(self):
		"""
		The method also initiates the cnvrg experiment with all its metrics.
		"""
		y_hat = self.__model.predict(self.__x_train)  # y_hat is a.k.a y_pred

		train_acc = accuracy_score(self.__y_train, y_hat)
		train_loss = mean_squared_error(self.__y_train, y_hat)

		y_pred = self.__model.predict(self.__x_test)
		test_acc = accuracy_score(self.__y_test, y_pred)
		test_loss = mean_squared_error(self.__y_test, y_pred)
		self.__metrics.update({
			'train_acc': train_acc,
			'train_loss': train_loss,
			'test_acc': test_acc,
			'test_loss': test_loss
		})
		self.__plot_all(y_pred)

	"""Plotting methods"""

	def __plot_feature_importance(self):
		"""Plots the feature importance."""
		try:
			if self.__testing_mode is False:
				self.__experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance', data=Bar(x=self.__features, y=self.__importance))
			else:
				print(self.__importance)
		except AttributeError:
			pass

	def __plot_classification_report(self, y_test_pred):
		"""Plots the classification report."""
		test_report = classification_report(self.__y_test, y_test_pred, output_dict=True)  # dict
		if self.__testing_mode is False:
			testing_report_as_array = self.__helper_plot_classification_report(test_report)
			self.__experiment.log_chart("Test Set - Classification Report", data=Heatmap(z=testing_report_as_array), y_ticks=self.__labels, x_ticks=["precision", "recall", "f1-score", "support"])
		else:
			print(test_report)

	def __plot_confusion_matrix(self, y_test_pred=None):
		""" Plots the confusion matrix. """
		if self.__y_test is not None and y_test_pred is not None:
			confusion_mat_test = confusion_matrix(self.__y_test, y_test_pred)  # array
			confusion_mat_test = SKTrainer.__helper_plot_confusion_matrix(confusion_mat_test)
			if self.__testing_mode is False:
				self.__experiment.log_chart("Test Set - confusion matrix", data=Heatmap(z=confusion_mat_test))
			else:
				print(confusion_mat_test)

	def __plot_roc_curve(self, y_test_pred):
		"""Plots the ROC curve."""
		true_values, false_values = [1, '1'], [0, '0']
		y_test = self.__y_test.tolist()
		y_pred = y_test_pred.tolist()
		if len(self.__labels) != 2 or self.__testing_mode is True or not (set(self.__labels) == {0, 1} or set(self.__labels) == {'0', '1'}):
			return

		TP, TN, FP, FN = 0, 0, 0, 0
		TPRs, FPRs = [0], [0]
		for ind in range(len(y_test)):
			if y_test[ind] in true_values:
				if y_pred[ind] in true_values: TP += 1
				else:                   FN += 1
			else:
				if y_pred[ind] in true_values: FP += 1
				else:                   TN += 1
			TPRs.append(TP / (TP + FN) if TP + FN != 0 else 0)
			FPRs.append(FP / (FP + TN) if FP + TN != 0 else 0)
		TPRs += [1]
		FPRs += [1]
		linearX, linearY = self.__plot_roc_curve_helper()
		self.__experiment.log_metric(key='ROC curve',
									 Ys=TPRs + linearY,
									 Xs=FPRs + linearX,
									 grouping=['roc'] * len(TPRs) + ['linear'] * len(linearX))


	def __plot_pandas_analyzer(self):
		"""Plots the cnvrg's pandas analyzer plots."""
		data = pd.concat([pd.concat([self.__x_train, self.__x_test], axis=0), pd.concat([self.__y_train, self.__y_test], axis=0)], axis=1)
		if self.__testing_mode is False:
			PandasAnalyzer(data, experiment=self.__experiment)

	def __plot_accuracies_and_errors(self):
		"""Plots the metrics."""
		self.__plot_accuracies_and_errors_helper()
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

		else:  # testing_mode is False
			self.__experiment.log_param("train set size", self.__train_set_size)
			self.__experiment.log_param("test set size", self.__test_set_size)
			self.__experiment.log_param("model", self.__metrics['model'])
			self.__experiment.log_param("test_acc", self.__metrics['test_acc'])
			self.__experiment.log_param("test_loss", self.__metrics['test_loss'])
			if self.__is_cross_val is True:
				self.__experiment.log_param("folds", self.__metrics['folds'])
				self.__experiment.log_metric("train_acc", self.__metrics['train_acc'], grouping=['train_acc'] * len(self.__metrics['train_acc']))
				self.__experiment.log_metric("train_loss", self.__metrics['train_loss'], grouping=['train_loss'] * len(self.__metrics['train_loss']))
				self.__experiment.log_metric("validation_acc", self.__metrics['validation_acc'], grouping=['validation_acc'] * len(self.__metrics['validation_acc']))
				self.__experiment.log_metric("validation_loss", self.__metrics['validation_loss'], grouping=['validation_loss'] * len(self.__metrics['validation_loss']))
				return
			self.__experiment.log_param("train_acc", self.__metrics['train_acc'])
			self.__experiment.log_param("train_loss", self.__metrics['train_loss'])

	def __plot_all(self, y_test_pred):
		# self.__plot_pandas_analyzer()
		# self.__plot_feature_importance()
		self.__plot_pandas_analyzer()
		self.__plot_classification_report(y_test_pred=y_test_pred)
		self.__plot_confusion_matrix(y_test_pred=y_test_pred)
		# self.__plot_roc_curve(y_test_pred=y_test_pred)
		self.__plot_accuracies_and_errors()

	def __save_model(self):
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + self.__output_model_name if os.environ.get("CNVRG_PROJECT_PATH") \
																									is not None else self.__output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))
		if not self.__testing_mode:
			os.system("ls -la {}".format(os.environ.get("CNVRG_PROJECT_PATH")))

	""" Helpers """
	@staticmethod
	def __helper_plot_confusion_matrix(confusion_matrix):
		""" Helper for plot_confusion_matrix. """
		output = []
		for y in range(len(confusion_matrix)):
			for x in range(len(confusion_matrix[y])):
				output.append((x, y, round(float(confusion_matrix[x][y]), SKTrainer.DIGITS_TO_ROUND)))
		return output

	def __plot_accuracies_and_errors_helper(self):
		"""Rounds all the values in self.__metrics"""
		keys_to_round = ['train_acc', 'train_loss', 'validation_acc', 'validation_loss', 'test_acc', 'test_loss']
		for key in keys_to_round:
			if key in self.__metrics.keys():
				if isinstance(self.__metrics[key], list) or isinstance(self.__metrics[key], numpy.ndarray):
					for ind in range(len(self.__metrics[key])):
						self.__metrics[key][ind] = round(self.__metrics[key][ind], SKTrainer.DIGITS_TO_ROUND)
				if isinstance(self.__metrics[key], numpy.ndarray):
					self.__metrics[key] = self.__metrics[key].tolist()
				else:
					self.__metrics[key] = round(self.__metrics[key], SKTrainer.DIGITS_TO_ROUND)

	def __plot_roc_curve_helper(self):
		""" Helper for the plot_roc_curve method. it creates the linear line values. """
		num_of_elements = len(self.__y_test)
		diff = 1 / num_of_elements
		x_axis, y_axis = [0], [0]
		for i in range(num_of_elements):
			x_axis.append(diff * i)
			y_axis.append(diff * i)
		x_axis.append(1)
		y_axis.append(1)
		return x_axis, y_axis

	def __helper_plot_classification_report(self, classification_report_dict):
		""" Converts dictionary given by classification_report to list of lists. """
		rows = []
		for k, v in classification_report_dict.items():
			if k in self.__labels:
				rows.append(list(v.values()))
		values = []
		for y in range(len(rows)):
			for x in range(len(rows[y])):
				values.append((x, y, round(rows[y][x], SKTrainer.DIGITS_TO_ROUND)))
		return values
