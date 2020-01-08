"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainer.py
==============================================================================
"""
import os
import pickle

import numpy as np
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar, Scatterplot
from cnvrg.charts.pandas_analyzer import PandasAnalyzer, MatrixHeatmap

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_squared_error


class SKTrainer:
	DIGITS_TO_ROUND = 3

	def __init__(self, model, train_set, test_set, output_model_name, testing_mode, folds=None):
		self.__model = model
		self.__x_train, self.__y_train = train_set
		self.__x_test, self.__y_test = test_set
		self.__testing_mode = testing_mode
		self.__cross_val_folds = folds
		self.__is_cross_val = (folds is not None)
		self.__features = list(self.__x_train.columns)
		self.__labels = [str(l) for l in list(set(self.__y_train).union(set(self.__y_test)))]
		self.__metrics = {'model': output_model_name,
						  'train set size': len(self.__y_train),
						  'test set size': len(self.__y_test)}
		self.__experiment = Experiment()

	def run(self):
		""" runs the training & testing methods. """
		self.__model.fit(self.__x_train, self.__y_train)

		if self.__is_cross_val:
			self.__metrics['folds'] = self.__cross_val_folds

		if self.__is_cross_val is True:
			self.__train_with_cross_validation()
		else:
			self.__train_without_cross_validation()
		self.__save_model()

	def __plot_all(self, y_test_pred):
		"""
		This method controls the visualization and metrics outputs.
		Hashtag something which you don't want to plot.
		"""
		self.__plot_correlation_matrix()
		self.__plot_feature_vs_feature()
		self.__plot_feature_importance()
		self.__plot_classification_report(y_test_pred=y_test_pred)
		self.__plot_confusion_matrix(y_test_pred=y_test_pred)
		self.__plot_roc_curve(y_test_pred=y_test_pred)
		self.__plot_accuracies_and_errors()

	"""training & testing methods"""

	def __train_with_cross_validation(self):
		"""
		This method enables sk-learn algorithms to perform KFold-cross-validation.
		The method also initiates the cnvrg experiment with all its metrics.
		"""
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

	def __plot_feature_importance(self):
		try:
			importance = getattr(self.__model, "feature_importances_")
			if self.__testing_mode is False:
				self.__experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance', data=Bar(x=self.__features, y=importance))
			else:
				print(importance)
		except AttributeError:
			pass

	def __plot_classification_report(self, y_test_pred):
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
		if len(set(self.__y_test)) != 2: return

		fpr, tpr, _ = roc_curve(self.__y_test, y_test_pred)
		if self.__testing_mode is False:
			self.__experiment.log_metric(key='ROC curve', Ys=tpr.tolist(), Xs=fpr.tolist())
		else: print("FPRs: {fpr}\nTPRs: {tpr}".format(fpr=fpr, tpr=tpr))

	def __plot_correlation_matrix(self):
		data = pd.concat([pd.concat([self.__x_train, self.__x_test], axis=0), pd.concat([self.__y_train, self.__y_test], axis=0)], axis=1)
		correlation = data.corr()
		self.__experiment.log_chart("correlation", [MatrixHeatmap(np.round(correlation.values, 2))],
									x_ticks=correlation.index.tolist(), y_ticks=correlation.index.tolist())

	def __plot_feature_vs_feature(self):
		data = pd.concat([pd.concat([self.__x_train, self.__x_test], axis=0), pd.concat([self.__y_train, self.__y_test], axis=0)], axis=1)
		indexes = data.select_dtypes(include=["number"]).columns
		corr = data.corr()
		for idx, i in enumerate(indexes):
			for jdx, j in enumerate(indexes):
				if i == j: continue
				if jdx < idx: continue
				corr_val = abs(corr[i][j])
				if 1 == corr_val or corr_val < 0.5: continue
				print("create", i, "against", j, "scatter chart")
				droplines = data[[i, j]].notnull().all(1)
				x, y = data[droplines][[i, j]].values.transpose()
				self.__experiment.log_chart("{i}_against_{j}".format(i=i, j=j),
											[Scatterplot(x=x.tolist(), y=y.tolist())],
											title="{i} against {j}".format(i=i, j=j))

	def __plot_accuracies_and_errors(self):
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
			for p in ['model', 'test_acc', 'test_loss', 'train set size', 'test set size']: self.__experiment.log_param(p, self.__metrics[p])
			if self.__is_cross_val is True:
				self.__experiment.log_param("folds", self.__metrics['folds'])
				metrics = ['train_acc', 'train_loss', 'validation_acc', 'validation_loss']
				for m in metrics: self.__experiment.log_metric(m, self.__metrics[m], grouping=[m] * len(self.__metrics[m]))
				return
			self.__experiment.log_param("train_acc", self.__metrics['train_acc'])
			self.__experiment.log_param("train_loss", self.__metrics['train_loss'])

	def __save_model(self):
		output_model_name = self.__metrics['model']
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") is not None else output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))

	""" --- Helpers --- """

	@staticmethod
	def __helper_plot_confusion_matrix(confusion_matrix):
		output = []
		for y in range(len(confusion_matrix)):
			for x in range(len(confusion_matrix[y])):
				output.append((x, y, round(float(confusion_matrix[x][y]), SKTrainer.DIGITS_TO_ROUND)))
		return output

	def __plot_accuracies_and_errors_helper(self):
		keys_to_round = ['train_acc', 'train_loss', 'validation_acc', 'validation_loss', 'test_acc', 'test_loss']
		for key in keys_to_round:
			if key in self.__metrics.keys():
				if isinstance(self.__metrics[key], list) or isinstance(self.__metrics[key], np.ndarray):
					for ind in range(len(self.__metrics[key])):
						self.__metrics[key][ind] = round(self.__metrics[key][ind], SKTrainer.DIGITS_TO_ROUND)
				if isinstance(self.__metrics[key], np.ndarray):
					self.__metrics[key] = self.__metrics[key].tolist()
				else:
					self.__metrics[key] = round(self.__metrics[key], SKTrainer.DIGITS_TO_ROUND)

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
