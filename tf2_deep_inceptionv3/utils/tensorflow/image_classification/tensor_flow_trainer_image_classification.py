"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Written by: Omer Liberman

Last update: Jun 01, 2020
Updated by: Omer Liberman

tensor_flow_trainer_image_classification.py
==============================================================================
"""
import os
import time
import cnvrg
import numpy as np
import tensorflow as tf

from cnvrg import Experiment
from cnvrg.charts import Heatmap
from sklearn.metrics import confusion_matrix

from utils.tensorflow.image_classification.model_generator import ModelGenerator
from utils.tensorflow.image_classification.tensor_trainer_utils import parse_classes, \
	load_generator, TimeHistory, export_labels_dictionary_from_classes_list

tf.compat.v1.disable_eager_execution()


"""
TensorFlowTrainerImageClassification
-----------------------
This module is used for training tensor-flow transfer learning modules
for image classification.

It runs the entire training and testing phases.
At the end, it logs metrics and parameters to cnvrg experiment, and 
save a trained model.
"""
class TensorFlowTrainerImageClassification:
	GRAYSCALE_CHANNELS = 1
	RGB_CHANNELS = 3
	VERBOSE = 1
	WORKERS = 3
	fully_connected_layers = [1024, 512, 256]

	METRICS = {
		'pre-training': [
						'TensorFlow version',
						'GPUs found',
						'Model',
						# 'Classes list'
					],
		'post-training': [
						'training_time',
						# 'epochs_duration',
						# 'avg_time_per_epoch',
						# 'time_per_step'
					],
		'post-test': [
					'test_acc',
					'test_loss'
				]
	}

	def __init__(self, arguments, model_name, base_model):
		"""
		Inits the trainer module.
		:param arguments: the arguments tuned by the user.
		:param model_name: the name of the output file.
		:param base_model: tensorflow model object.
		"""
		self.__arguments = arguments
		self.__shape = (arguments.image_height, arguments.image_width)
		self.__classes = parse_classes(arguments.data)
		self.__channels = TensorFlowTrainerImageClassification.RGB_CHANNELS if arguments.image_color == 'rgb' \
			else TensorFlowTrainerImageClassification.GRAYSCALE_CHANNELS
		self.__digits_to_round = arguments.digits_to_round
		self.__model = ModelGenerator(
									base_model=base_model,
									num_of_classes=len(self.__classes),
									fully_connected_layers=TensorFlowTrainerImageClassification.fully_connected_layers,
									loss_function=arguments.loss,
									dropout=arguments.dropout,
									activation_hidden_layers=arguments.hidden_layer_activation,
									activation_output_layers=arguments.output_layer_activation,
									optimizer=arguments.optimizer).get_model()

		tf_version = tf.__version__
		num_of_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

		self.__metrics = {
						'TensorFlow version': (str, tf_version),
						'GPUs found': (int, num_of_gpu),
						'Model': (str, model_name),
						'Classes list': (list, self.__classes),
						'digits_to_round': (int, self.__digits_to_round)}

		try:
			self.__cnvrg_env = True
			self.__experiment = Experiment()
		except:
			self.__cnvrg_env = False

	def run(self):
		"""
		This method runs the whole process:
		training, validation, testing and model's saving.

		:return: none.
		"""
		self.__plot(phase_in_process='pre-training')
		self.__train()
		self.__plot(phase_in_process='post-training')
		self.__test()
		self.__plot(phase_in_process='post-test')
		self.__save_model()

	def __plot(self, phase_in_process):
		"""
		This method controls the visualization and metrics plots.
		:param phase_in_process: the phase in the process.

		:return: none
		"""
		if phase_in_process == 'pre-training':
			self.__log_accuracies_and_losses(phase='pre-training')

		elif phase_in_process == 'post-training':
			self.__log_accuracies_and_losses(phase='post-training')

		elif phase_in_process == 'post-test' and self.__arguments.data_test is not None:
			self.__log_accuracies_and_losses(phase='post-test')
			self.__plot_confusion_matrix(self.__labels, self.__predictions)

	def __train(self):
		"""
		This method performs tensorflow algorithms training.
		The method also initiates the cnvrg experiment with all its metrics.
		:return: none.
		"""
		train_generator, val_generator = load_generator(
													self.__arguments.data,
													self.__shape,
													self.__arguments.test_size,  # test_size = validation_split
													self.__arguments.image_color,
													self.__arguments.batch_size)
		start_time = time.time()
		time_callback = TimeHistory()

		from PIL import ImageFile
		ImageFile.LOAD_TRUNCATED_IMAGES = True

		self.__model.fit(
			train_generator,
			epochs=self.__arguments.epochs,
			verbose=self.__arguments.verbose,
			steps_per_epoch=self.__arguments.steps_per_epoch,
			validation_data=val_generator if self.__arguments.test_size != 0. else None,
			validation_steps=self.__arguments.steps_per_epoch if self.__arguments.test_size != 0. else None,
			callbacks=[time_callback])

		training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		avg_time_per_epoch = round(sum(time_callback.times) / len(time_callback.times), self.__digits_to_round)

		self.__metrics['training_time'] = (str, training_time)
		self.__metrics['epochs_duration'] = (list, time_callback.times)
		self.__metrics['avg_time_per_epoch'] = (float, avg_time_per_epoch)

		if self.__arguments.steps_per_epoch is not None:
			time_per_step = [round(time_callback.times[i] / self.__arguments.steps_per_epoch, self.__digits_to_round)
							 for i in range(self.__arguments.epochs)]
			self.__metrics['time_per_step'] = (list, time_per_step)


	def __test(self):
		"""
		This method performs tensorflow algorithms testing.
		The method also initiates the cnvrg experiment with all its metrics.
		:return: none.
		"""
		# no data is given for testing.
		if self.__arguments.data_test is None:
			return

		test_gen = load_generator(
			self.__arguments.data_test,
			self.__shape,
			image_color=self.__arguments.image_color,
			batch_size=self.__arguments.batch_size,
			generate_test_set=True)

		self.__predictions = np.argmax(self.__model.predict(test_gen), axis=1)
		self.__labels = test_gen.classes

		steps_per_epoch_testing = test_gen.n
		test_loss, test_acc = self.__model.evaluate_generator(
															test_gen,
															workers=TensorFlowTrainerImageClassification.WORKERS,
															verbose=TensorFlowTrainerImageClassification.VERBOSE,
															steps=steps_per_epoch_testing)

		test_acc = (float, round(float(test_acc), self.__digits_to_round))
		test_loss = (float, round(float(test_loss), self.__digits_to_round))

		self.__metrics['test_acc'] = test_acc
		self.__metrics['test_loss'] = test_loss


	def __log_accuracies_and_losses(self, phase):
		"""
		This method logs the accuracies and the losses.
		If a cnvrg environment is recognized, it produces metrics and params at cnvrg experiment.
		Otherwise it prints the values to stdin.

		Note: it rounds the values in the table to 2 decimal digits.
		Otherwise, the UI seems broken.

		:return: none.
		"""
		metrics_for_current_phase = TensorFlowTrainerImageClassification.METRICS[phase]

		for metric in metrics_for_current_phase:
			# Uninitialized metric.
			if self.__metrics[metric] is None or metric not in self.__metrics:
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

	def __plot_confusion_matrix(self, y_test, y_test_pred):
		"""
		This method creates the confusion matrix.
		If a cnvrg environment is recognized, it produces a chart at cnvrg experiment.
		Otherwise it prints the values to stdin.

		:param y_test the true labels of the test set.
		:param y_test_pred: the predicted values for the test set.
		:return: none.
		"""
		confusion_matrix_test_set = confusion_matrix(y_test, y_test_pred)  # array
		confusion_matrix_test_set = self.__helper_plot_confusion_matrix(confusion_matrix_test_set)

		if self.__cnvrg_env:
			self.__experiment.log_chart("Test Set - confusion matrix", data=Heatmap(z=confusion_matrix_test_set))
		else:
			print('---Confusion Matrix---')
			print(confusion_matrix_test_set)
			print('----------------------')

	def __helper_plot_confusion_matrix(self, conf_matrix):
		"""
		This method is an helper for '''self.__plot_confusion_matrix()'''.
		It gets the conf_matrix of sk-learn and prepares it for structure required
		for cnvrg Heatmap feature (triplets of x,y,z).

		:param conf_matrix: the output of '''sklearn.metrics.confusion_matrix'''.
		:return: an array of triplets (x,y,z) where x,y are the coordinates in the table
		and z is the value.
		"""
		output = []
		for y in range(len(conf_matrix)):
			for x in range(len(conf_matrix[y])):
				z = round(float(conf_matrix[x][y]), self.__digits_to_round)
				output.append((x, y, z))
		return output

	def __save_model(self):
		"""
		This method saves the trained model object.
		If a cnvrg environment is recognized, it saves the model in the project directory.
		Otherwise, it saves the model in the path given at '''self.__metrics['output_model_file']'''.

		:return: none.
		"""
		if self.__cnvrg_env:
			dir_name = os.environ.get("CNVRG_WORKDIR") + "/" if os.environ.get("CNVRG_WORKDIR") is not None else ""
			file_name = self.__arguments.output_model
		else:
			dir_name, file_name = os.path.split(self.__arguments.output_model)
		self.__model.save(dir_name + '/' + file_name)
		export_labels_dictionary_from_classes_list(self.__classes, dir_name)