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
from cnvrg.charts import Bar, MatrixHeatmap, Scatterplot

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error


class SKTrainerRegression:
	DIGITS_TO_ROUND = 3

	REGRESSION_TYPE = ['linear', 'logistic']

	def __init__(self, model, train_set, test_set, output_model_name, testing_mode, folds=None, regression_type=0):
		self.__model = model
		self.__x_train, self.__y_train = train_set
		self.__train_set_size = len(self.__y_train)
		self.__x_test, self.__y_test = test_set
		self.__test_set_size = len(self.__y_test)
		self.__testing_mode = testing_mode
		self.__cross_val_folds = folds
		self.__is_cross_val = (folds is not None)
		self.__features = list(self.__x_train.columns)
		self.__labels = [str(l) for l in list(set(self.__y_train).union(set(self.__y_test)))]
		self.__metrics = {'model': output_model_name}
		self.__y_pred = None
		self.__experiment = Experiment.init('test_charts')  # replace with: self.__experiment = Experiment()
		self.__regression_type = SKTrainerRegression.REGRESSION_TYPE[regression_type]

		self.__coef, self.__intercept = None, None

	def run(self):
		self.__model.fit(self.__x_train, self.__y_train)

		try: self.__coef = self.__model.coef_
		except AttributeError: pass

		try: self.__intercept = self.__model.intercept_
		except AttributeError: pass

		if self.__is_cross_val:
			self.__metrics['folds'] = self.__cross_val_folds

		if self.__is_cross_val is True:
			self.__train_with_cross_validation()
		else:
			self.__train_without_cross_validation()
		self.__save_model()

	def __plot_all(self, y_test_pred):
		self.__plot_accuracies_and_errors()
		# self.__plot_regression_function()
		self.__plot_feature_importance()
		self.__plot_correlation_matrix()
		# self.__plot_feature_vs_feature()

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
			'test_loss_mse': test_loss})
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
			'test_loss_r2': test_loss_R2})
		self.__plot_all(self.__y_pred)

	def __plot_regression_function(self):
		if self.__regression_type == 'linear':
			a, b = self.__coef[0], self.__intercept
			x = np.linspace(-100, 100, 200)
			y = a * x + b
		elif self.__regression_type == 'logistic':
			x = np.linspace(-100, 100, 200)
			y = 1 / (1 + np.exp(-x))
		self.__experiment.log_metric(key="Regression Function", Xs=x.tolist(), Ys=y.tolist(), grouping=['regression line'] * len(x))

	def __plot_feature_importance(self):
		try:
			importance = getattr(self.__model, "feature_importances_")
			if self.__testing_mode is False:
				self.__experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance', data=Bar(x=self.__features, y=importance))
			else:
				print(importance)
		except AttributeError:
			pass

	def __plot_accuracies_and_errors(self):
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
		output_model_name = self.__metrics['model']
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") \
																				is not None else output_model_name
		pickle.dump(self.__model, open(output_file_name, 'wb'))

	"""training & testing methods"""

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