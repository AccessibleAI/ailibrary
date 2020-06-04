"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Last update: Oct 06, 2019
Updated by: Omer Liberman

base_model.py
==============================================================================
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, CosineSimilarity, \
	Huber, MeanAbsoluteError, MeanSquaredError


class ModelGenerator:
	def __init__(self,
				 base_model,
				 num_of_classes,
				 fully_connected_layers,
				 loss_function,
				 dropout=0.3,
				 activation_hidden_layers='relu',
				 activation_output_layers='softmax',
				 optimizer='adam'):
		self.__base_model = base_model
		self.__num_of_classes = num_of_classes
		self.__fully_connected_layers = fully_connected_layers
		self.__dropout = dropout
		self.__hidden_activation = activation_hidden_layers
		self.__output_activation =activation_output_layers
		self.__loss_function = ModelGenerator.__get_loss(loss_function, self.__num_of_classes)
		self.__optimizer = ModelGenerator.__get_optimizer(optimizer)
		self.__model = ModelGenerator.__create_model(self.__base_model,
													 self.__num_of_classes,
													 self.__fully_connected_layers,
													 self.__loss_function,
													 self.__dropout,
													 self.__hidden_activation,
													 self.__output_activation,
													 self.__optimizer)

	def get_model(self):
		return self.__model

	@staticmethod
	def __create_model(base_model, num_of_classes, fully_connected_layers, loss_function, dropout,
					   activation_func_hidden_layers, activation_func_output_layer, optimizer):
		for layer in base_model.layers:
			layer.trainable = False

		x = base_model.output
		x = GlobalAveragePooling2D()(x)

		if fully_connected_layers is not None:
			for layer in fully_connected_layers:
				x = Dense(layer, activation=activation_func_hidden_layers)(x)
				x = Dropout(dropout)(x)

		predictions_layer = Dense(num_of_classes, activation=activation_func_output_layer)(x)
		model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions_layer)
		model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
		return model

	@staticmethod
	def __get_optimizer(optimizer_name):
		if optimizer_name == 'sgd':       return tf.keras.optimizers.SGD()
		elif optimizer_name == 'rmsprop': return tf.keras.optimizers.RMSprop()
		elif optimizer_name == 'adagrad': return tf.keras.optimizers.Adagrad()
		elif optimizer_name == 'adam':    return tf.keras.optimizers.Adam()
		else:                             raise ValueError("AILibraryError: This optimizer is not available.")

	@staticmethod
	def __get_loss(loss, num_of_classes):
		if loss == 'cross_entropy':  # default value.
			return CategoricalCrossentropy() # if num_of_classes != 2 else BinaryCrossentropy()
		elif loss == 'binary_cross_entropy"':
			return BinaryCrossentropy()
		elif loss == 'cosine_similarity':
			return CosineSimilarity()
		elif loss == 'mean_absolute_error':
			return MeanAbsoluteError()
		elif loss == 'mean_squared_error':
			return MeanSquaredError()
		elif loss == 'huber':
			return Huber()
		else: raise ValueError('loss type does not exist.')

