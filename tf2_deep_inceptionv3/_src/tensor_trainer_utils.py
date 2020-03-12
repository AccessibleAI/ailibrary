"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

types.py
==============================================================================
"""
import os
import json
import tensorflow as tf

def cast_input_types(args):
	args.data_test = None if args.data_test == 'None' else args.data_test
	args.test_size = float(args.test_size)
	args.epochs = int(args.epochs)
	args.steps_per_epoch = int(args.steps_per_epoch)
	args.batch_size = int(args.batch_size)
	args.image_width = int(args.image_width)
	args.image_height = int(args.image_height)
	args.conv_width = int(args.conv_width)
	args.conv_height = int(args.conv_height)
	args.pool_width = int(args.pool_width)
	args.pool_height = int(args.pool_height)
	args.dropout = float(args.dropout)
	return args

def export_labels_dictionary_from_classes_list(classes):
	with open('labels.json', 'w') as fp:
		json.dump(classes, fp)

def export_labels_dictionary_from_generator(generator):
	classes = generator.class_indices
	with open('labels.json', 'w') as fp:
		json.dump(classes, fp)

def parse_classes(top_dir):
	subdirs = os.listdir(top_dir)
	classes = []
	for subdir in subdirs:
		if not subdir.startswith('.'): classes.append(subdir)
	return classes

def load_generator(data, image_size, val_size=0.2, image_color='rgb', batch_size=256, generate_test_set=False):
	"""
	Facade method.
	If the user wants to generate just a test set, insert data and set '''generate_test_set=True'''
	"""
	classes = parse_classes(data)
	class_mode = 'categorical' # if len(classes) != 2 else 'binary'

	if generate_test_set is True:
		return load_test_generator(data, image_size, image_color, batch_size, class_mode, classes)
	else:
		return load_train_and_val_generators(data, image_size, val_size, image_color, batch_size, class_mode, classes)

def load_train_and_val_generators(data, image_size, val_size, image_color, batch_size, class_mode, classes):
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


def load_test_generator(data, image_size, image_color, batch_size, class_mode, classes):
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
