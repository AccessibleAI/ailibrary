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
	GRAYSCALE_CHANNELS, RGB_CHANNELS = 1, 3
	VERBOSE = 1
	WORKERS = 3
	fully_connected_layers = [1024, 512, 256]

	def __init__(self, arguments, model_name, base_model):
		self.__cnvrg_env = True
		self.__arguments = cast_input_types(arguments)
		self.__shape = (arguments.image_height, arguments.image_width)
		self.__classes = parse_classes(arguments.data)
		self.__channels = TensorflowTrainer.RGB_CHANNELS if arguments.image_color == 'rgb' \
			else TensorflowTrainer.GRAYSCALE_CHANNELS
		self.__model = ModelGenerator(base_model=base_model,
					   num_of_classes=len(self.__classes),
					   fully_connected_layers=TensorflowTrainer.fully_connected_layers,
					   loss_function=arguments.loss,
					   dropout=arguments.dropout,
					   activation_hidden_layers=arguments.hidden_layer_activation,
					   activation_output_layers=arguments.output_layer_activation,
					   optimizer=arguments.optimizer).get_model()
		try: self.__experiment = Experiment()
		except cnvrg.modules.UserError: self.__cnvrg_env = False
		self.__metrics = {'tensorflow local version': tf.__version__,
						  'GPUs found': len(tf.config.experimental.list_physical_devices('GPU')),
						  'Model': model_name,
						  'Classes list': self.__classes}

	def run(self):
		if self.__cnvrg_env: self.__plot_all(status='pre-training')   ### using cnvrg.
		self.__train()
		self.__test()
		if self.__cnvrg_env:
			self.__plot_all()    ### using cnvrg.
			self.__export_model()    ### using cnvrg.

	def __plot_all(self, status='post-test'):
		if status == 'pre-training':
			self.__plot_metrics(status='pre-training')
		elif status == 'post-test' and self.__arguments.data_test is not None:
			self.__plot_metrics(status='post-test')
			self.__plot_confusion_matrix(self.__labels, self.__predictions)

	def __train(self):
		train_generator, val_generator = load_generator(self.__arguments.data, self.__shape,
														self.__arguments.test_size, self.__arguments.image_color,
														self.__arguments.batch_size)

		steps_per_epoch_training = self.__arguments.steps_per_epoch
		steps_per_epoch_validation = self.__arguments.steps_per_epoch

		start_time = time.time()
		time_callback = TimeHistory()

		print("---start training---")
		self.__model.fit(train_generator,
						epochs=self.__arguments.epochs,
						workers=multiprocessing.cpu_count() - 1,
						verbose=TensorflowTrainer.VERBOSE,
						steps_per_epoch=steps_per_epoch_training,
						validation_data=val_generator,
						validation_steps=steps_per_epoch_validation,
						use_multiprocessing=True,
						callbacks=[time_callback])
		print("---End training---")

		training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		self.__metrics['training_time'] = training_time

		if self.__cnvrg_env:
			self.__experiment.log_metric(key="Epoch Times", Ys=time_callback.times, Xs=[i for i in range(1, self.__arguments.epochs + 1)],
										x_axis="Epoch", y_axis="Time (Seconds)")

	def __test(self):
		if self.__arguments.data_test is None:
			return
		test_gen = load_generator(self.__arguments.data_test, self.__shape, image_color=self.__arguments.image_color,
								  batch_size=self.__arguments.batch_size, generate_test_set=True)
		self.__predictions = np.argmax(self.__model.predict(test_gen), axis=1)
		self.__labels = test_gen.classes

		steps_per_epoch_testing = test_gen.n
		test_loss, test_acc = self.__model.evaluate_generator(test_gen, workers=TensorflowTrainer.WORKERS,
															  verbose=TensorflowTrainer.VERBOSE, steps=steps_per_epoch_testing)
		test_acc, test_loss = round(float(test_acc), 3), round(float(test_loss), 3)
		self.__metrics['test_acc'] = test_acc
		self.__metrics['test_loss'] = test_loss

	def __export_model(self):
		output_file_name = os.environ.get("CNVRG_WORKDIR") + "/" + self.__arguments.output_model if os.environ.get("CNVRG_WORKDIR") is not None \
			else self.__arguments.output_model
		self.__model.save(output_file_name)
		export_labels_dictionary_from_classes_list(self.__classes)

	""" Cnvrg metrics output """
	def __plot_metrics(self, status='pre-training'):
		"""
		:param training_status: (String) either 'pre' or 'post'.
		"""
		if status == 'pre-training':
			print('Plotting pre-training metrics:')
			for k, v in self.__metrics.items():
				if k not in ['test_acc', 'test_loss']:
					self.__experiment.log_param(k, v)
		elif status == 'post-test':
			print('Plotting post-test metrics:')
			for k, v in self.__metrics.items():
				if k in ['test_acc', 'test_loss']:
					self.__experiment.log_param(k, v)
		else: raise ValueError('Unrecognized status.')

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

