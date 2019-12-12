"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

cnvrg_sklearn_helper.py
-----------------------
This file performs training with or without cross-validation over SK-learn models.
==============================================================================
"""
import os
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, roc_curve

from cnvrg import Experiment
from cnvrg.charts import Bar as Barchart, Heatmap, Scatterplot

import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# -----

# experiment = Experiment()
experiment = Experiment.init("test_charts")


def _plot_feature_importance(testing_mode, feature_names, importance):
	"""
	:param feature_names: the names of the features.
	:param importance: the model.feature_importances_
	:param testing_mode: boolean. for cnvrg inner test.
	:return:
	"""
	global experiment
	if testing_mode is False:
		experiment.log_chart('Feature Importance', x_axis='Features', y_axis='Importance', data = Barchart(x=feature_names, y=importance))
	else:  # Testing.
		print(importance)

def __helper_plot_classification_report(classification_report_dict, labels):
	"""
	Converts dictionary given by classification_report to list of lists.
	"""
	rows = []
	for k, v in classification_report_dict.items():
		if k in labels:
			rows.append(list(v.values()))
	values = []
	for y in range(len(rows)):
		for x in range(len(rows[y])):
			values.append((x, y, rows[y][x]))
	return values

def _plot_classification_report(testing_mode, y_train=None, y_train_pred=None, y_test=None, y_test_pred=None):
	global experiment
	labels = set(y_train) if y_train is not None else set(y_test)
	labels = [str(l) for l in labels]

	# if y_train is not None and y_train_pred is not None:
	# 	training_report = classification_report(y_train, y_train_pred, output_dict=True)  # dict
	# 	if testing_mode is False:
	# 		training_report_as_array = __helper_plot_classification_report(training_report, labels)
	# 		experiment.log_chart("Training Set - classification report", data=Heatmap(z=training_report_as_array), y_ticks=labels, x_ticks=["precision", "recall", "f1-score", "support"])
	# 	else:
	# 		print(training_report)

	if y_test is not None and y_test_pred is not None:
		test_report = classification_report(y_test, y_test_pred, output_dict=True)  # dict
		if testing_mode is False:
			testing_report_as_array = __helper_plot_classification_report(test_report, labels)
			experiment.log_chart("Testing Set - classification report", data=Heatmap(z=testing_report_as_array), y_ticks=labels, x_ticks=["precision", "recall", "f1-score", "support"])
		else:
			print(test_report)

def __helper_plot_confusion_matrix(confusion_matrix):
	output = []
	for y in range(len(confusion_matrix)):
		for x in range(len(confusion_matrix[y])):
			output.append((x, y, float(confusion_matrix[x][y])))
	return output

def _plot_confusion_matrix(testing_mode, y_train=None, y_train_pred=None, y_test=None, y_test_pred=None):
	global experiment

	# if y_train is not None and y_train_pred is not None:
	# 	confusion_mat_training = confusion_matrix(y_train, y_train_pred)  # array
	# 	parsed_confusion = __helper_plot_confusion_matrix(confusion_mat_training)
	# 	if testing_mode is False:
	# 		experiment.log_chart("Training Set - confusion matrix", data=Heatmap(z=parsed_confusion))
	# 	else:
	# 		print(confusion_mat_training)

	if y_test is not None and y_test_pred is not None:
		confusion_mat_test = confusion_matrix(y_test, y_test_pred)  # array
		confusion_mat_test = __helper_plot_confusion_matrix(confusion_mat_test)
		if testing_mode is False:
			experiment.log_chart("Test Set - confusion matrix", data=Heatmap(z=confusion_mat_test))
		else:
			print(confusion_mat_test)

def _plot_roc_curve(testing_mode, y_test, y_test_pred):
	global experiment
	classes = set(y_test)

	if len(classes) != 2:
		return

	fpr, tpr, _ = roc_curve(y_test, y_test_pred)
	if testing_mode is False:
		experiment.log_metric(key='ROC curve', Ys=tpr.tolist(), Xs=fpr.tolist())
	else:
		print("FPRs: ", fpr)
		print("TPRs: ", tpr)


def _plot_accuracies_and_errors(testing_mode, cross_validation, params_dict):
	global experiment
	if testing_mode is True:
		print("Model: {model}\n"
			  "train_acc={train_acc}\n"
			  "train_loss={train_loss}\n"
			  "test_acc={test_acc}\n"
			  "test_loss={test_loss}".format(
			model=params_dict['model'], train_acc=params_dict['train_acc'], train_loss=params_dict['train_loss'],
			test_acc=params_dict['test_acc'], test_loss=params_dict['test_loss']))
		if cross_validation is True:
			print("Folds: {folds}\n".format(folds=params_dict['folds']))

	if testing_mode is False:  # testing_mode is False
		experiment.log_param("model", params_dict['model'])
		experiment.log_param("test_acc", params_dict['test_acc'])
		experiment.log_param("test_loss", params_dict['test_loss'])
		if cross_validation is True:
			experiment.log_param("folds", params_dict['folds'])
			experiment.log_metric("train_acc", params_dict['train_acc'])
			experiment.log_metric("train_loss", params_dict['train_loss'])
			return
		experiment.log_param("train_acc", params_dict['train_acc'])
		experiment.log_param("train_loss", params_dict['train_loss'])


def _save_model(testing_mode, model_object, output_model_name):
	output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + output_model_name if os.environ.get("CNVRG_PROJECT_PATH") is not None else output_model_name
	pickle.dump(model_object, open(output_file_name, 'wb'))

	if not testing_mode:
		os.system("ls -la {}".format(os.environ.get("CNVRG_PROJECT_PATH")))


def train_with_cross_validation(model, train_set, test_set, folds, project_dir, output_model_name, testing_mode):
	"""
	This method enables sklearn algorithms to perform KFold-cross-validation.
	The method also initates the cnvrg.io experiment with all its metrics.
	:param model: SKlearn model object (initiated).
	:param train_set: tuple. (X_train, y_train). This is going to be used as a training set.
	:param test_set: tuple. (X_test, y_test). This is going to be used as a test set.
	:param folds: number of splits in the cross validation.
	:param project_dir: (Deprecared) the path to the directory which indicates where to save the model.
	:param output_model_name: the name of the output model saved on the disk.
	:param testing_mode: boolean. for cnvrg inner testing.
	:return: nothing.
	"""
	train_acc, train_loss = [], []
	kf = KFold(n_splits=folds)
	X, y = train_set

	model.fit(X, y)
	importance = model.feature_importances_

	# --- Training.
	for train_index, val_index in kf.split(X):
		X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
		y_train, y_val = y.iloc[train_index], y.iloc[val_index]

		model = model.fit(X_train, y_train)

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

	_plot_feature_importance(testing_mode, X.columns, importance)
	_plot_classification_report(testing_mode, y_test=y_test, y_test_pred=y_pred)
	_plot_confusion_matrix(testing_mode, y_test=y_test, y_test_pred=y_pred)
	_plot_roc_curve(testing_mode, y_test=y_test, y_test_pred=y_pred)

	params_dict = {
		'model': output_model_name,
		'folds': folds,
		'train_acc': train_acc,
		'train_loss': train_loss,
		'test_acc': test_acc,
		'test_loss': test_loss
	}
	_plot_accuracies_and_errors(testing_mode=testing_mode, cross_validation=True, params_dict=params_dict)

	# Save model.
	_save_model(testing_mode, model, output_model_name)


def train_without_cross_validation(model, train_set, test_set, project_dir, output_model_name, testing_mode):
	"""
	The method also initates the cnvrg.io experiment with all its metrics.
	:param model: SKlearn model object (initiated).
	:param train_set: tuple. (X_train, y_train). This is going to be used as a training set.
	:param test_set: tuple. (X_test, y_test). This is going to be used as a test set.
	:param project_dir: (Deprecared) the path to the directory which indicates where to save the model.
	:param output_model_name: the name of the output model saved on the disk.
	:param testing_mode: boolean. for cnvrg inner testing.
	:return: nothing.
	"""
	X_train, y_train = train_set

	# --- Training.
	model.fit(X_train, y_train)

	# Plot feature importance map.
	importance = model.feature_importances_

	y_hat = model.predict(X_train)  # y_hat is a.k.a y_pred

	train_acc = accuracy_score(y_train, y_hat)
	train_loss = mean_squared_error(y_train, y_hat)

	# --- Testing.
	X_test, y_test = test_set
	y_pred = model.predict(X_test)
	test_acc = accuracy_score(y_test, y_pred)
	test_loss = mean_squared_error(y_test, y_pred)

	_plot_feature_importance(testing_mode, X_train.columns, importance)
	_plot_classification_report(testing_mode, y_train=y_train, y_train_pred=y_hat, y_test=y_test, y_test_pred=y_pred)
	_plot_confusion_matrix(testing_mode, y_train=y_train, y_train_pred=y_hat, y_test=y_test, y_test_pred=y_pred)
	_plot_roc_curve(testing_mode, y_test=y_test, y_test_pred=y_pred)

	params_dict = {
		'model': output_model_name,
		'train_acc': train_acc,
		'train_loss': train_loss,
		'test_acc': test_acc,
		'test_loss': test_loss
	}
	_plot_accuracies_and_errors(testing_mode=testing_mode, cross_validation=False, params_dict=params_dict)

	# Save model.
	_save_model(testing_mode, model, output_model_name)

