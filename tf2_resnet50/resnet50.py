"""
All rights reserved to cnvrg.io
     http://www.cnvrg.io

cnvrg.io - AI library

Created by: Omer Liberman

Last update: Oct 19, 2019
Updated by: Omer Liberman

resnet50.py
==============================================================================
"""
import argparse

from src.training import training

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""ResNet50 Model""")

	parser.add_argument('--data', action='store', dest='data', required=True,
						help="""(String) (Required param) Path to a local directory which contains sub-directories, each for a single class. The data is used for training and validation.""")

	parser.add_argument('--data_test', action='store', dest='data_test', default=None,
						help="""(String) (Default: None) Path to a local directory which contains sub-directories, each for a single class. The data is used for testing.""")

	parser.add_argument('--project_dir', action='store', dest='project_dir',
	                    help="""String. (String) cnvrg.io parameter. NOT used by the user!""")

	parser.add_argument('--output_dir', action='store', dest='output_dir',
	                    help="""String. (String) cnvrg.io parameter. NOT used by the user!""")

	parser.add_argument('--output_model', action='store', default="model.h5", dest='output_model',
						help="""(String) (Default: 'model.h5') The name of the output model file. It is recommended to use '.h5' file.""")

	parser.add_argument('--test_size', action='store', default="0.2", dest='test_size',
	                    help="""(float) (Default: 0.2) The size of the validation / test set. If test set supplied, it represents the size of the validation set out of the data set given in --data. Otherwise, it represents the size of the test set out of the data set given in --data.""")

	parser.add_argument('--epochs', action='store', default="1", dest='epochs',
						help="""(int) (Default: 1) The number of epochs the algorithm performs in the training phase.""")

	parser.add_argument('--batch_size', action='store', default="256", dest='batch_size',
						help="""(int) (Default: 256) The number of images the generator downloads in each step.""")

	parser.add_argument('--image_color', action='store', dest='image_color', default='rgb',
	                    help="""(String) (Default: 'rgb') The colors of the images. Can be one of: 'grayscale', 'rgb'.""")

	parser.add_argument('--optimizer', action='store', dest='optimizer', default='adam',
	                    help="""(String) (Default: 'adam') The optimizer the algorithm uses. Can be one of: 'adam', 'adagrad', 'rmsprop', 'sgd'.""")

	parser.add_argument('--image_width', action='store', default="200", dest='image_width',
						help="""(int) (Default: 200) The width of the images.""")

	parser.add_argument('--image_height', action='store', default="200", dest='image_height',
						help="""(int) (Default: 200) The height of the images.""")

	parser.add_argument('--conv_width', action='store', default="3", dest='conv_width',
						help="""(int) (Default: 3) The width of the convolution window.""")

	parser.add_argument('--conv_height', action='store', default="3", dest='conv_height',
						help="""(int) (Default: 3) The height of the convolution window.""")

	parser.add_argument('--pool_width', action='store', default="2", dest='pool_width',
						help="""(int) (Default: 2) The width of the pooling window.""")

	parser.add_argument('--pool_height', action='store', default="2", dest='pool_height',
						help="""(int) (Default: 2) The height of the pooling window.""")

	parser.add_argument('--hidden_layer_activation', action='store', default='relu', dest='hidden_layer_activation',
						help="""(String) (Default: 'relu') The activation function of the hidden layers.""")

	parser.add_argument('--output_layer_activation', action='store', default='softmax', dest='output_layer_activation',
						help="""(String) (Default: 'softmax') The activation function of the output layer.""")

	args = parser.parse_args()

	train(args)
