"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Created by: Omer Liberman

Last update: Jan 27th, 2020
Updated by: Omer Liberman

inceptionv3.py
==============================================================================
"""
import argparse
import tensorflow as tf

from _src.tensor_trainer_utils import *
from _src.tensorflow_trainer import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""Mobilenet Model""")

	parser.add_argument('--data', action='store', dest='data', required=True,
						help="""(String) (Required param) Path to a local directory which contains sub-directories, each for a single class. The data is used for training and validation.""")

	parser.add_argument('--data_test', action='store', dest='data_test', default=None,
						help="""(String) (Default: None) Path to a local directory which contains sub-directories, each for a single class. The data is used for testing.""")

	parser.add_argument('--project_dir', action='store', dest='project_dir',
						help="""String. (String) cnvrg.io parameter. NOT used by the user!""")

	parser.add_argument('--output_dir', action='store', dest='output_dir',
						help="""String. (String) cnvrg.io parameter. NOT used by the user!""")

	parser.add_argument('--test_mode', action='store', default=False, dest='test_mode',
						help="""--- For inner use of cnvrg.io ---""")

	parser.add_argument('--output_model', action='store', default="model.h5", dest='output_model',
						help="""(String) (Default: 'model.h5') The name of the output model file. It is recommended to use '.h5' file.""")

	parser.add_argument('--validation_split', action='store', default="0.", dest='test_size',
						help="""(float) (Default: 0.) The size of the validation set. If test set supplied, it represents the size of the validation set 
						out of the data set given in --data. Otherwise, it represents the size of the test set out of the data set given in --data.""")

	parser.add_argument('--epochs', action='store', default="1", dest='epochs',
						help="""(int) (Default: 1) The number of epochs the algorithm performs in the training phase.""")

	parser.add_argument('--steps_per_epoch', action='store', default="None",
						help="""(int or None) (Default: None) If its None -> num of samples / batch_size, otherwise -> The number of batches done in each epoch.""")

	parser.add_argument('--batch_size', action='store', default="32", dest='batch_size',
						help="""(int) (Default: 32) The number of images the generator downloads in each step.""")

	parser.add_argument('--workers', action='store', default="1", dest='workers',
						help="""(int) (Default: 1) The number of workers which are used.""")

	parser.add_argument('--multi_processing', action='store', default="False", dest='multi_processing',
						help="""(boolean) (Default: False) Indicates whether to run multi processing.""")

	parser.add_argument('--verbose', action='store', default='1', dest='verbose',
						help="""(integer) (Default: 1) can be either 1 or 0.""")

	parser.add_argument('--image_color', action='store', dest='image_color', default='rgb',
						help="""(String) (Default: 'rgb') The colors of the images. Can be one of: 'grayscale', 'rgb'.""")

	parser.add_argument('--loss', action='store', dest='loss', default='cross_entropy',
						help="""(String) (Default: 'crossentropy') The loss function of the model. By default its binary_crossentropy or categorical_crossentropy, 
						depended of the classes number.""")

	parser.add_argument('--dropout', action='store', dest='dropout', default='0.3',
						help="""(float) (Default: 0.3) The dropout of the added fully connected layers.""")

	parser.add_argument('--optimizer', action='store', dest='optimizer', default='adam',
						help="""(String) (Default: 'adam') The optimizer the algorithm uses. Can be one of: 'adam', 'adagrad', 'rmsprop', 'sgd'.""")

	parser.add_argument('--image_width', action='store', default="256", dest='image_width',
						help="""(int) (Default: 256) The width of the images.""")

	parser.add_argument('--image_height', action='store', default="256", dest='image_height',
						help="""(int) (Default: 256) The height of the images.""")

	parser.add_argument('--conv_width', action='store', default="3", dest='conv_width',
						help="""(int) (Default: 3) The width of the convolution window.""")

	parser.add_argument('--conv_height', action='store', default="3", dest='conv_height',
						help="""(int) (Default: 3) The height of the convolution window.""")

	parser.add_argument('--pooling_width', action='store', default="2", dest='pool_width',
						help="""(int) (Default: 2) The width of the pooling window.""")

	parser.add_argument('--pooling_height', action='store', default="2", dest='pool_height',
						help="""(int) (Default: 2) The height of the pooling window.""")

	parser.add_argument('--hidden_layer_activation', action='store', default='relu', dest='hidden_layer_activation',
						help="""(String) (Default: 'relu') The activation function of the hidden layers.""")

	parser.add_argument('--output_layer_activation', action='store', default='softmax', dest='output_layer_activation',
						help="""(String) (Default: 'softmax') The activation function of the output layer.""")

	args = parser.parse_args()

	args = cast_input_types(args)
	channels = 3 if args.image_color == 'rgb' else 1
	base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False,
												input_shape=(args.image_height, args.image_width, channels))
	trainer = TensorflowTrainer(args, 'MobileNet', base_model)
	trainer.run()