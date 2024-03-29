"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io
TensorflowTrainer.py
==============================================================================
"""
import multiprocessing
import os
import time
import cnvrg
import numpy as np
import tensorflow as tf

from cnvrg import Experiment
from cnvrg.charts import Heatmap
from sklearn.metrics import confusion_matrix

from _src.tensor_trainer_utils import *
from _src.time_history import TimeHistory
from _src.model_generator import ModelGenerator

tf.compat.v1.disable_eager_execution()


class TensorflowTrainer:
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
		self.__cnvrg_env = True
		self.__arguments = arguments
		self.__shape = (arguments.image_height, arguments.image_width)
		self.__classes = parse_classes(arguments.data)
		self.__channels = TensorflowTrainer.RGB_CHANNELS if arguments.image_color == 'rgb' \
			else TensorflowTrainer.GRAYSCALE_CHANNELS
		self.__model = ModelGenerator(
									base_model=base_model,
									num_of_classes=len(self.__classes),
									fully_connected_layers=TensorflowTrainer.fully_connected_layers,
									loss_function=arguments.loss,
									dropout=arguments.dropout,
									activation_hidden_layers=arguments.hidden_layer_activation,
									activation_output_layers=arguments.output_layer_activation,
									optimizer=arguments.optimizer).get_model()
		try:
			print("Trying to launch an experiment in cnvrg environment.")
			self.__experiment = Experiment()
		except Exception:
			print("Not in cnvrg environment.")
			self.__cnvrg_env = False

		self.__metrics = {
						'TensorFlow version': tf.__version__,
						'GPUs found': len(tf.config.experimental.list_physical_devices('GPU')),
						'Model': model_name,
						'Classes list': self.__classes}

	def run(self):
		self.__plot(status='pre-training')

		self.__train()
		self.__plot(status='post-training')

		self.__test()
		self.__plot(status='post-test')

		self.__export_model()

	def __plot(self, status):
		if status == 'pre-training':
			self.__plot_metrics(status='pre-training')

		elif status == 'post-training':
			self.__plot_metrics(status='post-training')

		elif status == 'post-test' and self.__arguments.data_test is not None:
			self.__plot_metrics(status='post-test')
			self.__plot_confusion_matrix(self.__labels, self.__predictions)

	def __train(self):
		train_generator, val_generator = load_generator(
													self.__arguments.data,
													self.__shape,
													self.__arguments.test_size,  # test_size = validation_split
													self.__arguments.image_color,
													self.__arguments.batch_size)

		start_time = time.time()
		time_callback = TimeHistory()

		print("--- Starts Training ---")

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

		print("--- Ends training ---")

		training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		self.__metrics['training_time'] = training_time
		self.__metrics['epochs_duration'] = Metric(key='Epochs Duration', Ys=time_callback.times, Xs='from_1', x_axis='epochs', y_axis='time (seconds)')
		self.__metrics['avg_time_per_epoch'] = round(sum(time_callback.times) / len(time_callback.times), 3)

		if self.__arguments.steps_per_epoch is not None:
			self.__metrics['time_per_step'] = Metric(
													key='Time per Step',
													Ys=[round(time_callback.times[i] / self.__arguments.steps_per_epoch, 3) for i in range(self.__arguments.epochs)],
													Xs='from_1', x_axis='epochs',
													y_axis='time (ms)/step')

	def __test(self):
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
															workers=TensorflowTrainer.WORKERS,
															verbose=TensorflowTrainer.VERBOSE,
															steps=steps_per_epoch_testing)

		test_acc, test_loss = round(float(test_acc), 3), round(float(test_loss), 3)
		self.__metrics['test_acc'] = test_acc
		self.__metrics['test_loss'] = test_loss

	def __export_model(self):
		output_file_name = os.environ.get("CNVRG_WORKDIR") + "/" + self.__arguments.output_model if os.environ.get("CNVRG_WORKDIR") is not None \
			else self.__arguments.output_model
		self.__model.save(output_file_name)
		export_labels_dictionary_from_classes_list(self.__classes)

	# ============ Helpers ============

	def __plot_metrics(self, status):
		metrics = TensorflowTrainer.METRICS[status]

		if status == 'pre-training':
			for metric in metrics:
				if self.__cnvrg_env:
					if metric in self.__metrics.keys():  # if metric exists
						self.__experiment.log_param(metric, self.__metrics[metric])
				else:
					print("log_param -  {key} : {value}".format(key=metric, value=self.__metrics[metric]))

		elif status == 'post-training':
			for metric in metrics:
				if metric in self.__metrics.keys():        # if metric exists
					if not isinstance(self.__metrics[metric], Metric):   # param
						if self.__cnvrg_env:
							self.__experiment.log_param(metric, self.__metrics[metric])
						else:
							print("log_param -  {key} : {value}".format(key=metric, value=self.__metrics[metric]))
					else:   # metrics should be called here.
						if self.__cnvrg_env:
							self.__experiment.log_metric(
														key=self.__metrics[metric].key,
														Ys=self.__metrics[metric].Ys,
														Xs=self.__metrics[metric].Xs,
														x_axis=self.__metrics[metric].x_axis,
														y_axis=self.__metrics[metric].y_axis)
						else:
							print(self.__metrics[metric])

		elif status == 'post-test':
			for metric in metrics:
				if metric in self.__metrics.keys():  # if metric exists
						
					if self.__cnvrg_env:
						self.__experiment.log_param(metric, self.__metrics[metric])
					else:
						print("log_param -  {key} : {value}".format(key=metric, value=self.__metrics[metric]))

		else:
			raise ValueError('Unrecognized status.')

	def __plot_confusion_matrix(self, labels, predictions):
		""" Plots the confusion matrix. """
		confusion_mat_test = confusion_matrix(labels, predictions)  # array
		confusion_mat_test = TensorflowTrainer.__helper_plot_confusion_matrix(confusion_mat_test, mat_x_ticks=self.__classes, mat_y_ticks=self.__classes)
		self.__experiment.log_chart("confusion matrix", data=Heatmap(z=confusion_mat_test))

	@staticmethod
	def __helper_plot_confusion_matrix(confusion_matrix, mat_x_ticks=None, mat_y_ticks=None, digits_to_round=3):
		"""
		:param confusion_matrix: the values in the matrix.
		:param mat_x_ticks, mat_y_ticks: ticks for the axis of the matrix.
		"""
		output = []
		for y in range(len(confusion_matrix)):
			for x in range(len(confusion_matrix[y])):
				x_val = x if mat_x_ticks is None else mat_x_ticks[x]
				y_val = y if mat_y_ticks is None else mat_y_ticks[y]
				output.append((x_val, y_val, round(float(confusion_matrix[x][y]), digits_to_round)))
		return output
