"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Last update: Oct 06, 2019
Updated by: Omer Liberman

cnvrg_base_model.py
==============================================================================
"""
import tensorflow as tf

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def _get_optimizer(optimizer_name):
	"""
	Returns an optimizer object.
	:param optimizer_name: string.
	:param learning_rate: float in [0, 1]
	:return: tensorflow.src.keras.optimizers object.
	"""
	if optimizer_name == 'sgd':
		return tf.keras.optimizers.SGD()
	elif optimizer_name == 'rmsprop':
		return tf.keras.optimizers.RMSprop()
	elif optimizer_name == 'adagrad':
		return tf.keras.optimizers.Adagrad()
	elif optimizer_name == 'adam':
		return tf.keras.optimizers.Adam()
	else:
		raise Exception("Unknown optimizer")


def init_model(base_model,
               num_of_classes,
               fully_connected_layers,
               activation_func_hidden_layers,
               activation_func_output_layer,
               optimizer='adam'):

	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)

	if fully_connected_layers is not None:
		for fc in fully_connected_layers:
			x = tf.keras.layers.Dense(fc, activation=activation_func_hidden_layers)(x)
			x = tf.keras.layers.Dropout(0.5)(x)

	predictions_layer = tf.keras.layers.Dense(num_of_classes, activation=activation_func_output_layer)(x)

	model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions_layer)

	model.compile(optimizer=_get_optimizer(optimizer), loss='categorical_crossentropy', metrics=['accuracy'])

	return model