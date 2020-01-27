ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. The fundamental breakthrough with ResNet is that it enables training extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.

## Notes for this library
The library is a wrapper for the ``Tensorflow2`` implementation of the ResNet50 module. The library enables the user to create a deep neural network based on ResNet50.

It can be used both for colored ('rgb') images and for black-white image ('grayscale') by modifying the parameter ```--image_color```.

The library also enables the user to load the images directly from the local directory. By default the library needs the receive a single path (``--data``) for a local directory. The library splits the content of the directory to train set and validation/test set according to the parameter ```--test_size``` which by default equals 0.2. If you are also interested in supplying a test set, it can be given through the parameter ```--data_test```. If you supply it, the library performs training, validation and testing.

## Parameters

```--data``` - str, required. Path to a local directory which contains sub-directories, each for a single class. The data is used for training and validation.

```--data_test``` - str (default = None). Path to a local directory which contains sub-directories, each for a single class. The data is used for testing. 

```--output_model``` - str (default = 'model.h5'). The name of the output model file. It is recommended to use the '.h5' filetype.

```--val_size``` - float (default = '0.2'). The size of the validation / test set. If a test set is supplied with the ```-data_test``` parameter, this ratio represents the size of the validation set created out of the data set given in --data. Otherwise, it represents the size of the test set out of the data set given in --data.

```--epochs``` - int (default = '3'). The number of epochs the algorithm performs in the training phase.

```--batch_size``` - int (default = '256'). The number of images the generator downloads in each step.

```--image_color``` - str (default = 'rgb') The colors of the images. Can be either: 'grayscale' or 'rgb'.

```--loss``` - str (Default = 'crossentropy'). The loss function of the model. By default it is 'categorical_crossentropy'.,

```--dropout``` - float (Default = '0.3'). The dropout of the added fully connected layers.

```--optimizer``` - str (Default = 'adam'). The optimizer the algorithm will use. Can be 'adam', 'adagrad', 'rmsprop' or 'sgd'.

```--image_height``` - int (Default = '224'). The height of the images in pixels.

```--image_width``` - int (Default = '224'). The width of the images in pixels.

```--conv_width``` - int (Default = '3'). The width of the convolution window in pixels.

```--conv_height``` - int (Default = '3'). The height of the convolution window in pixels.

```--pooling_width``` - int (Default = '2'). The width of the pooling window in pixels.

```--pooling_height``` - int (Default = '2'). The height of the pooling window in pixels.

```--hidden_layer_activation``` - str (Default = 'relu'). The activation function of the hidden layers.

```--output_layer_activation``` - str (Default = 'softmax'). The activation function of the output layer.