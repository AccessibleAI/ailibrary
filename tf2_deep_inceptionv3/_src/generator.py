"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

generator.py
==============================================================================
"""
import os
import json
import tensorflow as tf


def load_generator(data, image_size, val_size=0.2, image_color='rgb', batch_size=256, generate_test_set=False):
	"""
	Facade method.
	If the user wants to generate just a test set, insert data and set '''generate_test_set=True'''
	"""
	classes = parse_classes(data)
	class_mode = 'categorical' # if len(classes) != 2 else 'binary'

	if generate_test_set is True:
		return __load_test_generator(data, image_size, image_color, batch_size, class_mode, classes)
	else:
		return __load_train_and_val_generators(data, image_size, val_size, image_color, batch_size, class_mode, classes)


def export_labels_dictionary(generator):
	classes = generator.class_indices
	with open('labels.json', 'w') as fp:
		json.dump(classes, fp)


def parse_classes(top_dir):
	subdirs = os.listdir(top_dir)
	classes = []
	for subdir in subdirs:
		if not subdir.startswith('.'): classes.append(subdir)
	return classes


def __load_train_and_val_generators(data, image_size, val_size, image_color, batch_size, class_mode, classes):
	"""
	Returns two generators contains the files in the directory data..
	"""
	data_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=val_size)

	train_gen = data_gen.flow_from_directory(
		data,
		color_mode=image_color,
		target_size=image_size,
		batch_size=batch_size,
		class_mode=class_mode,
		subset='training',
		classes=classes,
		shuffle=True)

	val_gen = data_gen.flow_from_directory(
		data,
		color_mode=image_color,
		target_size=image_size,
		batch_size=batch_size,
		class_mode=class_mode,
		subset='validation',
		classes=classes,
		shuffle=True)

	return train_gen, val_gen


def __load_test_generator(data, image_size, image_color, batch_size, class_mode, classes):
	"""
	returns single generator for the test set.
	"""
	data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

	test_gen = data_gen.flow_from_directory(
		data,
		color_mode=image_color,
		target_size=image_size,
		batch_size=batch_size,
		class_mode=class_mode,
		classes=classes,
		shuffle=True)

	return test_gen
