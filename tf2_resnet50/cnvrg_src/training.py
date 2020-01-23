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
from cnvrg_src.casting import cast_types
from cnvrg_src.cnvrg_base_model import init_model
from cnvrg_src.cnvrg_images_generator import load_images_to_generators, output_generator_dictionary

VERBOSE = 1
WORKERS = 1
RGB_NUM_OF_COLORS = 3
GRAYSCALE_NUM_OF_COLORS = 1

tf.compat.v1.disable_eager_execution()

def train(args, model_name):
	args = cast_types(args)

	# Set basic params.
	input_shape = (args.image_height, args.image_width)
	num_of_colors = RGB_NUM_OF_COLORS if args.image_color == 'rgb' else GRAYSCALE_NUM_OF_COLORS

	# Get images generators.
	train_generator, val_generator = load_images_to_generators(data=args.data,
	                                                           batch_size=args.batch_size,
	                                                           image_size=input_shape,
	                                                           image_color=args.image_color,
	                                                           test_size=args.test_size)
	# Model's initiation.
	base_model = tf.keras.applications.ResNet50(weights='imagenet',
	                                            include_top=False,
	                                            input_shape=(args.image_height, args.image_width, num_of_colors))
	model = init_model(base_model=base_model,
	                   num_of_classes=len(set(train_generator.classes)),
	                   fully_connected_layers=[1024, 512, 256],
	                   activation_func_hidden_layers=args.hidden_layer_activation,
	                   activation_func_output_layer=args.output_layer_activation,
	                   optimizer=args.optimizer)
	# train.
	steps_per_epoch_training = train_generator.n // args.epochs
	steps_per_epoch_validation = val_generator.n // args.epochs
	start_time = time.time()
	train_metrics = model.fit_generator(train_generator,
										epochs=args.epochs,
										workers=WORKERS,
										verbose=VERBOSE,
										steps_per_epoch=steps_per_epoch_training,
										validation_data=val_generator,
										validation_steps=steps_per_epoch_validation)

	training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

	# test.
	test_acc, test_loss = None, None
	if args.data_test is not None:
		test_set_path = args.data_test
		test_gen = load_images_to_generators(data=test_set_path,
											  batch_size=args.batch_size,
											  image_size=input_shape,
											  image_color=args.image_color)
		steps_per_epoch_testing = test_gen.n
		test_loss, test_acc = model.evaluate_generator(test_gen,
		                                               workers=WORKERS,
		                                               verbose=VERBOSE,
		                                               steps=steps_per_epoch_testing)
		test_acc, test_loss = float(test_acc), float(test_loss)

	exp = Experiment()
	exp.log_param('Tensorflow version', tf.__version__)
	exp.log_param('GPUs available', len(tf.config.experimental.list_physical_devices('GPU')))
	exp.log_param("training_time", training_time)
	exp.log_param("model_name", model_name)

	if test_acc is not None and test_loss is not None:
		exp.log_param("test_acc", test_acc)
		exp.log_param("test_loss", test_loss)

	# Save.
	output_file_name = os.environ.get("CNVRG_PROJECT_PATH") + "/" + args.output_model if os.environ.get("CNVRG_PROJECT_PATH") is not None else args.output_model
	model.save(output_file_name)
	output_generator_dictionary(train_generator)
