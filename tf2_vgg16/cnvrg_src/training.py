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
from src.casting import cast_types
from src.cnvrg_base_model import init_model
from src.cnvrg_images_generator import load_images_to_generators, output_generator_dictionary

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

	output_generator_dictionary(train_generator)

	# Model's initiation.
	base_model = tf.keras.applications.VGG16(weights='imagenet',
	                                            include_top=False,
	                                            input_shape=(args.image_height, args.image_width, num_of_colors))

	model = init_model(base_model=base_model,
	                   num_of_classes=len(set(train_generator.classes)),
	                   fully_connected_layers=[1024, 1024, 512],
	                   activation_func_hidden_layers=args.hidden_layer_activation,
	                   activation_func_output_layer=args.output_layer_activation,
	                   optimizer=args.optimizer)

	# train.
	steps_per_epoch_training = train_generator.n // train_generator.batch_size

	start_time = time.time()
	train_metrics = model.fit_generator(train_generator,
	                                    epochs=args.epochs,
	                                    workers=WORKERS,
	                                    verbose=VERBOSE,
	                                    steps_per_epoch=steps_per_epoch_training)

	training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
	train_loss = train_metrics.history['loss']  # list
	train_acc = train_metrics.history['accuracy']  # list

	# test.
	steps_per_epoch_testing = val_generator.n // val_generator.batch_size
	test_loss, test_acc = model.evaluate_generator(val_generator,
	                                               workers=WORKERS,
	                                               verbose=VERBOSE,
	                                               steps=steps_per_epoch_testing)

	if not args.test_mode:
		# Initiating cnvrg.io experiment.
		exp = Experiment()
		exp.log_metric("train_loss", train_loss)
		exp.log_metric("train_acc", train_acc)
		exp.log_param("test_loss", test_loss)
		exp.log_param("test_acc", test_acc)
		exp.log_param("training_time", training_time)
		exp.log_param("model_name", model_name)
	else:
		print("Model: {model}\n"
			  "train_acc={train_acc}\n"
			  "train_loss={train_loss}\n"
			  "test_acc={test_acc}\n"
			  "test_loss={test_loss}\n"
			  "training_time={training_time}".format(
			model=model_name, training_time=training_time, train_acc=train_acc, train_loss=train_loss, test_acc=test_acc, test_loss=test_loss))

	# Save.
	output_file_name = os.environ['PROJECT_DIR'] + "/" + args.output_model if os.environ['PROJECT_DIR'] is not None else args.output_model
	model.save(output_file_name)