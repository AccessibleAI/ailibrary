"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

training.py
==============================================================================
"""
# For downloading ImageNet:
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import time
import tensorflow as tf

from cnvrg import Experiment
from _src.types import _cast
from _src.base_model import ModelGenerator
from _src.generator import load_generator, export_labels_dictionary, parse_classes

VERBOSE = 1
WORKERS = 1
GRAYSCALE_CHANNELS, RGB_CHANNELS = 1, 3

tf.compat.v1.disable_eager_execution()


def train_and_test(args, model_name):
	args = _cast(args)
	input_shape = (args.image_height, args.image_width)
	channels = RGB_CHANNELS if args.image_color == 'rgb' else GRAYSCALE_CHANNELS
	classes = parse_classes(args.data)

	base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(args.image_height, args.image_width, channels))

	model = ModelGenerator(base_model=base_model,
						   num_of_classes=len(classes),
						   fully_connected_layers=[1024, 512, 256],
						   loss_function=args.loss,
						   dropout=args.dropout,
						   activation_hidden_layers=args.hidden_layer_activation,
						   activation_output_layers=args.output_layer_activation,
						   optimizer=args.optimizer).get_model()

	exp = Experiment()
	exp.log_param('Tensorflow version', tf.__version__)
	exp.log_param('GPUs available', len(tf.config.experimental.list_physical_devices('GPU')))
	exp.log_param("Model Name", model_name)
	exp.log_param('Classes', classes)

	# ------ Training & Validating.
	train_generator, val_generator = load_generator(args.data, input_shape, args.test_size, args.image_color, args.batch_size)
	steps_per_epoch_training = train_generator.n // args.epochs
	steps_per_epoch_validation = val_generator.n // args.epochs
	start_time = time.time()
	model.fit_generator(train_generator,
						epochs=args.epochs,
						workers=WORKERS,
						verbose=VERBOSE,
						steps_per_epoch=steps_per_epoch_training,
						validation_data=val_generator,
						validation_steps=steps_per_epoch_validation)

	training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
	exp.log_param("training_time", training_time)

	# ------ Testing.
	if args.data_test is not None:
		test_gen = load_generator(data=args.data_test,
								  image_size=input_shape,
								  image_color=args.image_color,
								  batch_size=args.batch_size,
								  generate_test_set=True)
		steps_per_epoch_testing = test_gen.n
		test_loss, test_acc = model.evaluate_generator(test_gen,
													   workers=WORKERS,
													   verbose=VERBOSE,
													   steps=steps_per_epoch_testing)
		test_acc, test_loss = float(test_acc), float(test_loss)
		exp.log_param("test_acc", test_acc)
		exp.log_param("test_loss", test_loss)

	# ------ Saving.
	output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + args.output_model if os.environ.get("CNVRG_PROJECT_PATH") is not None else args.output_model
	model.save(output_file_name)
	export_labels_dictionary(train_generator)
