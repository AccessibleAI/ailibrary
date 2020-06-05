import itertools
import time
import unittest
import os
from itertools import combinations

CURRENT_DIR = os.getcwd()
SCRIPTS_DIR = CURRENT_DIR[:-len('_tests')]

DATA_DIR = CURRENT_DIR + '/data/'
OUTPUTS_DIR = CURRENT_DIR + '/outputs/'

DATA_PATH_SK_LEARN = DATA_DIR + 'test_data.csv'
SK_LEARN_LOSS_TYPES = ['F1', 'LOG', 'MSE', 'MAE', 'R2']
CROSS_VALIDATION = "10"

DATA_PATH_TENSOR_FLOW_TRAIN = DATA_DIR + 'images/training_set'
DATA_PATH_TENSOR_FLOW_TEST = DATA_DIR + 'images/test_set'
TENSORFLOW_VALIDATION_SPLITS = ['0.1']
TENSORFLOW_EPOCHS = ['10']
TENSORFLOW_LOSS = ['binary_cross_entropy', 'mean_absolute_error', 'mean_squared_error']
TENSORFLOW_OPTIMIZERS = ['sgd', 'rmsprop']
TENSORFLOW_ACTIVATIONS = ['relu']


# library name: (library type, directory name, library script name)
LIB_DIR_IND = 1
LIB_SCRIPT_INT = 2

sk_learn_libraries = {
	'decision-trees': ('sklearn_classification', 'sk_classification_DecisionTrees', 'decision_trees_classifier.py'),
	'gradient-boost': ('sklearn_classification', 'sk_classification_GradientBoosting', 'gradient_boosting.py'),
	'k-nearest-neighbors': ('sklearn_classification', 'sk_classification_KNN', 'knn.py'),
	'naive-bayes': ('sklearn_classification', 'sk_classification_NaiveBayes', 'naive_bayes.py'),
	'random-forest': ('sklearn-classification', 'sk_classification_RandomForest', 'random_forest_classifier.py'),
	'svm': ('sklearn-classification', 'sk_classification_SVM', 'svm.py'),
	'xgboost': ('sklearn-classification', 'xgboost', 'xgb.py'),
	'linear-regression': ('sklearn-regression', 'sk_regression_Linear', 'linear_regression.py'),
	'logistic-regression': ('sklearn-regression', 'sk_regression_Logistic', 'logistic_regression.py'),
	'random-forest-regression': ('sklearn-regression', 'sk_regression_RandomForestReg', 'random_forest_regressor.py'),
}

tensorflow_libraries = {
	'inceptionV3': ('tensorflow-classification', 'tf2_deep_inceptionv3', 'inceptionv3.py'),
	'mobilenNet': ('tensorflow-classification', 'tf2_deep_mobilenet', 'mobilenet.py'),
	'resnet50': ('tensorflow-classification', 'tf2_deep_resnet50', 'resnet50.py'),
	'vgg16': ('tensorflow-classification', 'tf2_deep_vgg16', 'vgg16.py')
}

def compose_command_sklearn(library_name, data, output_model, cross_validation="None", loss_type="MSE"):
	path = SCRIPTS_DIR + '/' + sk_learn_libraries[library_name][LIB_DIR_IND] + '/' + sk_learn_libraries[library_name][LIB_SCRIPT_INT]
	command = "python3 {path} --data={data} --x_val={x_val} " \
			  "--train_loss_type={loss} --test_loss_type={loss} " \
			  "--digits_to_round=4 --output_model={model}".format(path=path, data=data, model=output_model, x_val=cross_validation, loss=loss_type)
	return command


def compose_command_tensorflow(library_name, data, output_model, val_split, epochs,
							   loss, optimizers, activations, test_data="None"):
	path = SCRIPTS_DIR + '/' + tensorflow_libraries[library_name][LIB_DIR_IND] + '/' + tensorflow_libraries[library_name][LIB_SCRIPT_INT]
	command = "python3 {path} --data={data} --data_test={data_test} " \
			  "--digits_to_round=4 --output_model={model} " \
			  "--validation_split={val_split} --epochs={epochs} --loss={loss} " \
			  "--optimizer={optimizer} --hidden_layer_activation={activation} " \
			  "--output_layer_activation={activation}" \
			  "".format(path=path, data=data,
						model=output_model, data_test=test_data,
						val_split=val_split, epochs=epochs, loss=loss,
						optimizer=optimizers, activation=activations)
	return command


class SKLearnTest(unittest.TestCase):

	def test_decision_tress_no_cross_validation(self):
		lib = 'decision-trees'
		model = "decision_trees.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_decision_tress_with_cross_validation(self):
		lib = 'decision-trees'
		model = "decision_trees.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_gradient_boost_no_cross_validation(self):
		lib = 'gradient-boost'
		model = "gradient_boost.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_gradient_boost_with_cross_validation(self):
		lib = 'gradient-boost'
		model = "gradient_boost.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_knn_no_cross_validation(self):
		lib = 'k-nearest-neighbors'
		model = "knn.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_knn_with_cross_validation(self):
		lib = 'k-nearest-neighbors'
		model = "knn.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_naive_bayes_no_cross_validation(self):
		lib = 'naive-bayes'
		model = "nb.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_naive_bayes_with_cross_validation(self):
		lib = 'naive-bayes'
		model = "nb.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_random_forest_no_cross_validation(self):
		lib = 'random-forest'
		model = "rf.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_random_forest_with_cross_validation(self):
		lib = 'random-forest'
		model = "rf.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_svm_no_cross_validation(self):
		lib = 'svm'
		model = "svm.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_svm_with_cross_validation(self):
		lib = 'svm'
		model = "svm.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			print(command)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_linear_regression_no_cross_validation(self):
		lib = 'linear-regression'
		model = "lr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_linear_regression_with_cross_validation(self):
		lib = 'linear-regression'
		model = "lr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_logistic_regression_no_cross_validation(self):
		lib = 'logistic-regression'
		model = "logr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_logistic_regression_with_cross_validation(self):
		lib = 'logistic-regression'
		model = "logr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_rf_regression_no_cross_validation(self):
		lib = 'random-forest-regression'
		model = "rfr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_rf_regression_with_cross_validation(self):
		lib = 'random-forest-regression'
		model = "rfr.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_xgb_no_cross_validation(self):
		lib = 'xgboost'
		model = "xgb.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)

	def test_xgb_with_cross_validation(self):
		lib = 'xgboost'
		model = "xgb.sav"

		for loss in SK_LEARN_LOSS_TYPES:
			command = compose_command_sklearn(library_name=lib, data=DATA_PATH_SK_LEARN,
											  output_model=OUTPUTS_DIR + model, cross_validation=CROSS_VALIDATION,
											  loss_type=loss)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced with loss {}.".format(loss)
			os.remove(OUTPUTS_DIR + model)


class TensorflowTest(unittest.TestCase):

	def test_inception_no_test_data(self):
		lib = 'inceptionV3'
		model = "inception.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
								   loss, optimizer, activation, test_data="None")
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

	def test_inception_with_test_data(self):
		lib = 'inceptionV3'
		model = "inception.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
								   loss, optimizer, activation, test_data=DATA_PATH_TENSOR_FLOW_TEST)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

	def test_resnet_no_test_data(self):
		lib = 'resnet50'
		model = "resnet.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
												 loss, optimizer, activation, test_data="None")
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

	def test_resnet_with_test_data(self):
		lib = 'resnet50'
		model = "resnet.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
												 loss, optimizer, activation, test_data=DATA_PATH_TENSOR_FLOW_TEST)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

	def test_vgg_no_test_data(self):
		lib = 'vgg16'
		model = "vgg.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
												 loss, optimizer, activation, test_data="None")
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

	def test_vgg_with_test_data(self):
		lib = 'vgg16'
		model = "vgg.h5"

		for s in itertools.product(TENSORFLOW_VALIDATION_SPLITS, TENSORFLOW_EPOCHS, TENSORFLOW_LOSS,
								   TENSORFLOW_OPTIMIZERS, TENSORFLOW_ACTIVATIONS):
			val_split, epochs, loss, optimizer, activation = s
			command = compose_command_tensorflow(lib, DATA_PATH_TENSOR_FLOW_TRAIN, OUTPUTS_DIR + model, val_split, epochs,
												 loss, optimizer, activation, test_data=DATA_PATH_TENSOR_FLOW_TEST)
			os.system(command)
			assert model in os.listdir(OUTPUTS_DIR), "model file hasn't been produced."
			assert "labels.json" in os.listdir(OUTPUTS_DIR), "labels file hasn't been produced."
			os.remove(OUTPUTS_DIR + model)
			os.remove(OUTPUTS_DIR + 'labels.json')

if __name__ == '__main__':
	unittest.main()
