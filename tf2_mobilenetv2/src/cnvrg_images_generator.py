"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

cnvrg_images_generator.py
----------------
==============================================================================
"""
import json
import tensorflow as tf

def load_images_to_generators(data, image_size, test_size, image_color='rgb', batch_size=128):
	"""
	Returns two generators contains the files in the directory data.
	The amount of data is splited between the two generators by the val_size param.
	:param data: String. Path to the top directory.
	:param image_size: Tuple of integers. the size of the images.
	:param test_size: Float. in range (0, 1). valdiation set size.
	:param image_color: String. can be either 'rgb' or 'grayscale'.
	:param batch_size: Integer. size of a batch.
	:return: training_generator, validation_generator.
	"""
	data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
	                                                            validation_split=test_size)

	train_generator = data_gen.flow_from_directory(data,
	                                               color_mode=image_color,
	                                               target_size=image_size,
	                                               batch_size=batch_size,
	                                               class_mode='categorical',
	                                               subset='training',
	                                               shuffle=True)  # set as training data

	val_generator = data_gen.flow_from_directory(data,
	                                             color_mode=image_color,
	                                             target_size=image_size,
	                                             batch_size=batch_size,
	                                             class_mode='categorical',
	                                             subset='validation',
	                                             shuffle=True)  # set as validation data

	return train_generator, val_generator


def output_generator_dictionary(generator):
	classes = generator.class_indices
	with open('labels_dict.json', 'w') as fp:
		json.dump(classes, fp)



