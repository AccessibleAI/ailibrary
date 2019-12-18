"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

SKTrainer.py
==============================================================================
"""
import os
import pickle
import pandas as pd

from cnvrg import Experiment
from cnvrg.charts import Heatmap, Bar
from cnvrg.charts.pandas_analyzer import PandasAnalyzer

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, mean_squared_error


class SKTrainer:
	DIGITS_TO_ROUND = 3

	def __init__(self, model, train_set, test_set, output_model_name, testing_mode, folds=None):
		self.__model = model
		self.__x_train, self.__y_train = train_set
		self.__x_test, self.__y_test = test_set
		self.__output_model_name = output_model_name
		self.__testing_mode = testing_mode
		self.__cross_val_folds = folds
		self.__is_cross_val = (folds is not None)
		self.__features = list(self.__x_train.columns)
		self.__labels = [str(l) for l in list(set(self.__y_train).union(set(self.__y_test)))]

		self.__model.fit(self.__x_train, self.__y_train)
		self.__importance = self.__model.feature_importances_

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

	"""training & testing methods"""

	def __train_with_cross_validation(self):
		"""
		This method enables sk-learn algorithms to perform KFold-cross-validation.
		The method also initiates the cnvrg experiment with all its metrics.
		"""
		train_acc, train_loss = [], []
		kf = KFold(n_splits=self.__cross_val_folds)

		for train_index, val_index in kf.split(self.__x_train):
			X_train, X_val = self.__x_train.iloc[train_index, :], self.__x_train.iloc[val_index, :]
			y_train, y_val = self.__y_train.iloc[train_index], self.__y_train.iloc[val_index]
			self.__model = self.__model.fit(X_train, y_train)

			y_hat = self.__model.predict(X_val)  # y_hat is a.k.a y_pred
			acc = accuracy_score(y_val, y_hat)
			loss = mean_squared_error(y_val, y_hat)

			train_acc.append(acc)
			train_loss.append(loss)

		# --- Testing.
		y_pred = self.__model.predict(self.__x_test)
		test_acc = accuracy_score(self.__y_test, y_pred)
		test_loss = mean_squared_error(self.__y_test, y_pred)
		self.__metrics.update({
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
		if self.__testing_mode is False:
			self.__experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance', data=Bar(x=self.__features, y=self.__importance))
		else:
			print(self.__importance)

	def __plot_classification_report(self, y_test_pred):
		test_report = classification_report(self.__y_test, y_test_pred, output_dict=True)  # dict
		if self.__testing_mode is False:
			testing_report_as_array = self.__helper_plot_classification_report(test_report)
			self.__experiment.log_chart("Test Set - Classification Report", data=Heatmap(z=testing_report_as_array), y_ticks=self.__labels, x_ticks=["precision", "recall", "f1-score", "support"])
		else:
			print(test_report)

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

	def __plot_confusion_matrix(self, y_test_pred=None):
		if self.__y_test is not None and y_test_pred is not None:
			confusion_mat_test = confusion_matrix(self.__y_test, y_test_pred)  # array
			confusion_mat_test = self.__helper_plot_confusion_matrix(confusion_mat_test)
			if self.__testing_mode is False:
				self.__experiment.log_chart("Test Set - confusion matrix", data=Heatmap(z=confusion_mat_test))
			else:
				print(confusion_mat_test)

	def __helper_plot_confusion_matrix(self, confusion_matrix):
		output = []
		for y in range(len(confusion_matrix)):
			for x in range(len(confusion_matrix[y])):
				output.append((x, y, round(float(confusion_matrix[x][y]), SKTrainer.DIGITS_TO_ROUND)))
		return output

	def __plot_roc_curve(self, y_test_pred):
		n_classes = len(self.__labels)
		y_test = self.__y_test.tolist()
		y_test_pred = y_test_pred.tolist()
		if n_classes != 2 or self.__testing_mode is True:
			return
		y_test, y_test_pred = list(y_test), list(y_test_pred)
		FPRs, TPRs, _ = roc_curve(y_test, y_test_pred)
		self.__experiment.log_metric(key='ROC curve', Ys=TPRs.tolist(), Xs=FPRs.tolist())

	def __plot_pandas_analyzer(self):
		data = pd.concat([pd.concat([self.__x_train, self.__x_test], axis=0), pd.concat([self.__y_train, self.__y_test], axis=0)], axis=1)
		if self.__testing_mode is False:
			PandasAnalyzer(data, experiment=self.__experiment)

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
			self.__experiment.log_param("model", self.__metrics['model'])
			self.__experiment.log_param("test_acc", self.__metrics['test_acc'])
			self.__experiment.log_param("test_loss", self.__metrics['test_loss'])
			if self.__is_cross_val is True:
				self.__experiment.log_param("folds", self.__metrics['folds'])
				self.__experiment.log_metric("train_acc", self.__metrics['train_acc'])
				self.__experiment.log_metric("train_loss", self.__metrics['train_loss'])
				return
			self.__experiment.log_param("train_acc", self.__metrics['train_acc'])
			self.__experiment.log_param("train_loss", self.__metrics['train_loss'])

	def __plot_accuracies_and_errors_helper(self):
		"""Rounds all the values in self.__metrics"""
		keys_to_round = ['train_acc', 'train_loss', 'test_acc', 'test_loss']
		for key in keys_to_round:
			self.__metrics[key] = round(self.__metrics[key], SKTrainer.DIGITS_TO_ROUND)

	def __plot_all(self, y_test_pred):
		"""
		Runs all the plotting methods.
		"""
		self.__plot_pandas_analyzer()
		self.__plot_feature_importance()
		self.__plot_classification_report(y_test_pred=y_test_pred)
		self.__plot_confusion_matrix(y_test_pred=y_test_pred)
		self.__plot_roc_curve(y_test_pred=y_test_pred)
		self.__plot_accuracies_and_errors()
		self.__save_model()

	"""technical methods"""

	def __save_model(self):
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + self.__output_model_name if os.environ.get("CNVRG_PROJECT_PATH") \
																									is not None else self.__output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))
		if not self.__testing_mode:
			os.system("ls -la {}".format(os.environ.get("CNVRG_PROJECT_PATH")))



