"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

TensorTrainer.py
==============================================================================
"""
import os
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from cnvrg import Experiment
from cnvrg_src.ImagesLoader import ImagesLoader


class TensorTrainer:

	VERBOSE = 1
	WORKERS = 1
	RGB_NUM_OF_COLORS = 3
	GRAYSCALE_NUM_OF_COLORS = 1

	def __init__(self, base_model, model_name, args):
		self.__base_model = base_model
		self.__model_name = model_name
		self.__params = args

		self.__input_shape = (args.image_height, args.image_width)
		self.__colors = TensorTrainer.RGB_NUM_OF_COLORS if args.image_color == 'rgb' else TensorTrainer.GRAYSCALE_NUM_OF_COLORS

		self.__images_loader = ImagesLoader(data=args.data,
											image_size=self.__input_shape,
											validation_size=self.__params.test_size,
											image_color=self.__params.image_color,
											batch_size=args.batch_size)
		self.__train_gen, self.__val_gen = self.__images_loader.load()
		self.__num_of_classes = len(set(self.__train_gen.classes))

		self.__metrics = {}
		self.__experiment = Experiment.init("test_charts")

	def run(self):
		self.__create_model()
		self.__train()
		self.__plot_metrics()
		self.__save()

	def __create_model(self, fully_connected_layers=[1024, 1024, 512]):
		"""Prepares the final model layers and put it in self.__final_model."""
		num_of_classes = self.__num_of_classes
		for layer in self.__base_model.layers:
			layer.trainable = False
		x = self.__base_model.output
		x = tf.keras.layers.GlobalAveragePooling2D()(x)

		if fully_connected_layers is not None:
			for fc in fully_connected_layers:
				x = tf.keras.layers.Dense(fc, activation=self.__params.hidden_layer_activation)(x)
				x = tf.keras.layers.Dropout(0.5)(x)

		predictions_layer = tf.keras.layers.Dense(num_of_classes, activation=self.__params.output_layer_activation)(x)
		model = tf.keras.models.Model(inputs=self.__base_model.input, outputs=predictions_layer)

		model.compile(optimizer=self.__get_optimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

		self.__final_model = model


	def __train(self):
		"""Runs the actual training."""
		steps_per_epoch_training = self.__train_gen.n // self.__params.epochs
		start_time = time.time()
		train_metrics = self.__final_model.fit_generator(self.__train_gen,
														epochs=self.__params.epochs,
														workers=TensorTrainer.WORKERS,
														verbose=TensorTrainer.VERBOSE,
														steps_per_epoch=steps_per_epoch_training)
		training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		train_loss = train_metrics.history['loss']      # list
		train_acc = train_metrics.history['accuracy']   # list
		steps_per_epoch_testing = self.__val_gen.n // self.__params.epochs
		test_loss, test_acc = self.__final_model.evaluate_generator(self.__val_gen,
																	workers=TensorTrainer.WORKERS,
																	verbose=TensorTrainer.VERBOSE,
												 					steps=steps_per_epoch_testing)
		self.__metrics.update({
			'train_loss': train_loss,
			'train_acc': train_acc,
			'test_loss': test_loss,
			'test_acc': test_acc,
			'training_time': training_time}
		)

	def __plot_metrics(self):
		"""Plots the metrics."""
		if not self.__params.test_mode:
			self.__experiment.log_metric("train_loss", self.__metrics['train_loss'])
			self.__experiment.log_metric("train_acc", self.__metrics['train_acc'])
			self.__experiment.log_param("test_loss", self.__metrics['test_loss'])
			self.__experiment.log_param("test_acc", self.__metrics['test_acc'])
			self.__experiment.log_param("training_time", self.__metrics['training_time'])
			self.__experiment.log_param("model_name", self.__model_name)
		else:
			print("Model: {model}\n"
				  "train_acc={train_acc}\n"
				  "train_loss={train_loss}\n"
				  "test_acc={test_acc}\n"
				  "test_loss={test_loss}\n"
				  "training_time={training_time}".format(
				model=self.__model_name,
				training_time=self.__metrics['training_time'],
				train_acc=self.__metrics['train_acc'],
				train_loss=self.__metrics['train_loss'],
				test_acc=self.__metrics['test_acc'],
				test_loss=self.__metrics['test_loss']))

	def __save(self):
		# Save.
		output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + self.__params.output_model if os.environ.get("CNVRG_PROJECT_PATH") is not None else self.__params.output_model
		self.__final_model.save(output_file_name)
		# os.system('ls -la {}'.format(os.environ.get("CNVRG_PROJECT_PATH")))

	def __get_optimizer(self):
		"""Returns an optimizer object."""
		if self.__params.optimizer == 'sgd':
			return tf.keras.optimizers.SGD()
		elif self.__params.optimizer == 'rmsprop':
			return tf.keras.optimizers.RMSprop()
		elif self.__params.optimizer == 'adagrad':
			return tf.keras.optimizers.Adagrad()
		elif self.__params.optimizer == 'adam':
			return tf.keras.optimizers.Adam()
		else:
			raise Exception("Unknown optimizer")
