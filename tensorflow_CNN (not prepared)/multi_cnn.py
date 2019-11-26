"""
multi_cnn.py
---------
"""
import os
import argparse
import tensorflow as tf
from numpy import append, asarray
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical


def init_model(num_of_classes, img_width, img_height, conv_wind_width, conv_wind_height, pool_wind_width, pool_wind_height, activation_func_hidden_layers, activation_func_output_layer):
	"""
	img_width, img_height - sizes of the input image.
	conv_wind_width, conv_wind_height - sizes of the convolution window.
	pool_wind_width, pool_wind_height - sizes of the pooling window.
	activation_func_hidden_layers, activation_func_output_layer are the types of the activation funcs.
	"""
	conv_wind = (conv_wind_width, conv_wind_height)
	pool_wind = (pool_wind_width, pool_wind_height)

	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Conv2D(32, conv_wind, input_shape=(img_width, img_height, 3)))
	model.add(tf.keras.layers.Activation(activation_func_hidden_layers))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_wind))

	model.add(tf.keras.layers.Conv2D(32, conv_wind))
	model.add(tf.keras.layers.Activation(activation_func_hidden_layers))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_wind))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.Activation(activation_func_hidden_layers))
	model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(num_of_classes))
	model.add(tf.keras.layers.Activation(activation_func_output_layer))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model


def main(args):
	args = cast_types(args)

	# Loading data to numpy arrays.
	paths_to_classes = args.dir
	all_sub_folders = get_all_sub_folders(paths_to_classes)

	input_type = args.input_type
	input_shape = (args.img_width, args.img_height)

	X, y = None, None
	NUM_OF_CLASSES = len(all_sub_folders)

	for label in range(NUM_OF_CLASSES):
		X_curr, y_curr = load_images_from_path(path=all_sub_folders[label], input_type=input_type, input_shape=input_shape, label=label)

		if X is None and y is None:
			X, y = X_curr, y_curr
		else:
			X = append(X, X_curr, axis=0)
			y = append(y, y_curr)
		del X_curr
		del y_curr
	X /= 255
	y = to_categorical(y)

	# Split to train/test.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	del X
	del y

	# model initiation.
	model = init_model(
		len(all_sub_folders),
		args.img_width,
		args.img_height,
		args.conv_width,
		args.conv_height,
		args.pool_width,
		args.pool_height,
		args.hidden_activation,
		args.output_activation
	)

	BATCH_SIZE, EPOCHS = args.batch, args.epochs

	# train.
	hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
	loss_train, acc_train = hist.history['loss'], hist.history['acc']
	print('cnvrg_tag_loss_train:', loss_train)
	print('cnvrg_tag_accuracy_train:', acc_train)

	# test.
	score = model.evaluate(X_test, y_test)
	loss_test, acc_test = score[0], score[1]

	print('cnvrg_tag_loss_val:', loss_test)
	print('cnvrg_tag_accuracy_val:', acc_test)

	# Save.
	where_to_save = args.project_dir + "/" + args.model if args.project_dir is not None else args.model
	model.save(where_to_save)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""Deep Learning CNN Model for binary classification (images)""")

	parser.add_argument('--top_dir', action='store', dest='dir', required=True,
						help="""Array of strings. path to the folders of the images of all classes. """)

	parser.add_argument('--input_type', action='store', dest='input_type', default=".jpg",
						help="The type of the images.")

	parser.add_argument('--project_dir', action='store', dest='project_dir', help="""String.. """)

	parser.add_argument('--output_dir', action='store', dest='output_dir', help="""String.. """)

	parser.add_argument('--model', action='store', default="CNN_Model.h5", dest='model',
						help="""String. The name of the output file which is a trained model """)

	parser.add_argument('--epochs', action='store', default="20", dest='epochs',
						help="""Num of epochs. Default is 20""")

	parser.add_argument('--batch', action='store', default="128", dest='batch', help="""batch size. Default is 128""")

	parser.add_argument('--img_width', action='store', default="200", dest='img_width',
						help=""" The width of the input images .Default is 200""")

	parser.add_argument('--img_height', action='store', default="200", dest='img_height',
						help=""" The height of the input images .Default is 200""")

	parser.add_argument('--conv_width', action='store', default="3", dest='conv_width',
						help=""" The width of the convolution window.Default is 3""")

	parser.add_argument('--conv_height', action='store', default="3", dest='conv_height',
						help=""" The height of the convolution window.Default is 3""")

	parser.add_argument('--pool_width', action='store', default="2", dest='pool_width',
						help=""" The width of the pooling window.Default is 3""")

	parser.add_argument('--pool_height', action='store', default="2", dest='pool_height',
						help=""" The height of the pooling window.Default is 3""")

	parser.add_argument('--hidden_activation', action='store', default='relu', dest='hidden_activation',
						help="""The activation function for the hidden layers.""")

	parser.add_argument('--output_activation', action='store', default='softmax', dest='output_activation',
						help="""The activation function for the output layer.""")

	args = parser.parse_args()

	main(args)
