"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

ImagesLoader.py
==============================================================================
"""
import json
import tensorflow as tf

class ImagesLoader:
	"""
	Loads images from given path to 2 python data generator.
	The given directory should look like:
	- path
		- class 1 dir
		- class 2 dir
		- ...
		- class n dir.
	"""
	def __init__(self, data, image_size, validation_size, image_color='rgb', batch_size=128):
		self.__data = data                     # (String)
		self.__image_size = image_size         # tuple(int, int)
		self.__val_size = validation_size      # float
		self.__color = image_color             # 'rgb'/ 'gray'
		self.__batch_size = batch_size         # integer.

	def load(self):
		"""Main method."""
		self.__create_generators()
		self.__create_dictionary_of_labels_and_numbers(save=True)
		return self.__train_generator, self.__val_generator

	def __create_generators(self):
		"""Creates the training and validation generators."""
		data_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=self.__val_size)
		self.__train_generator = data_gen.flow_from_directory(self.__data,
													   color_mode=self.__color,
													   target_size=self.__image_size,
													   batch_size=self.__batch_size,
													   class_mode='categorical',
													   subset='training',
													   shuffle=True)
		self.__val_generator = data_gen.flow_from_directory(self.__data,
													   color_mode=self.__color,
													   target_size=self.__image_size,
													   batch_size=self.__batch_size,
													   class_mode='categorical',
													   subset='validation',
													   shuffle=True)

	def __create_dictionary_of_labels_and_numbers(self, save):
		"""Creates a json file with a dictionary looks like {numerical_label: original_label}.
		   This json file is used for prediction."""
		self.__labels_dict = self.__train_generator.class_indices

		if save is True:
			with open('labels.json', 'w') as fp:
				json.dump(self.__labels_dict, fp)
