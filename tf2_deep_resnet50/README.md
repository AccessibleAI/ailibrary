ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. The fundamental breakthrough with ResNet is that it enables training extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.

## Notes for this library
The library is a wrapper for the ``Tensorflow2`` implementation of the ResNet50 module. The library enables the user to create a deep neural network based on ResNet50.

It can be used both for colored ('rgb') images and for black-white image ('grayscale') by modifying the parameter ```--image_color```.

The library also enables the user to load the images directly from the local directory. By default the library needs the receive a single path (``--data``) for a local directory. The library splits the content of the directory to train set and validation/test set according to the parameter ```--test_size``` which by default equals 0.2. If you are also interested in supplying a test set, it can be given through the parameter ```--data_test```. If you supply it, the library performs training, validation and testing.

## Parameters

```--data``` - (String) (Required param) Path to a local directory which contains sub-directories, each for a single class. The data is used for training and validation.

```--data_test``` - (String) (Default: None) Path to a local directory which contains sub-directories, each for a single class. The data is used for testing. 

```--output_model``` - (String) (Default: 'model.h5') The name of the output model file. It is recommended to use '.h5' file.

```--validation_split``` - (float) (Default: 0.) The size of the validation / test set. If test set supplied, it represents the size of the validation set out of the data 
set given in --data. Otherwise, it represents the size of the test set out of the data set given in --data.

```--epochs``` - (int) (Default: 1) The number of epochs the algorithm performs in the training phase.

```--batch_size``` - (int) (Default: 256) The number of images the generator downloads in each step.

```--steps_per_epoch``` - (int or None) (Default: None) If its None -> num of samples / batch_size, otherwise -> The number of batches done in each epoch.

```--batch_size``` - (int) (Default: 32) The number of images the generator downloads in each step.

```--workers``` - (int) (Default: 1) The number of workers which are used.

```--multi_processing``` - (boolean) (Default: False) Indicates whether to run multi processing.

```--verbose``` - (integer) (Default: 1) can be either 1 or 0.

```--image_color``` - (String) (Default: 'rgb') The colors of the images. Can be one of: 'grayscale', 'rgb'.

```--loss``` - (String) (Default: 'crossentropy') The loss function of the model. By default its binary_crossentropy or categorical_crossentropy, depended of the classes number.

```--dropout``` - (float) (Default: 0.3) The dropout of the added fully connected layers.

```--optimizer``` - (String) (Default: 'adam') The optimizer the algorithm uses. Can be one of: 'adam', 'adagrad', 'rmsprop', 'sgd'.

```--image_height``` - (int) (Default: 256) The height of the images.

```--image_width``` - (int) (Default: 256) The width of the images.

```--conv_width``` - (int) (Default: 3) The width of the convolution window.

```--conv_height``` - (int) (Default: 3) The height of the convolution window.

```--pooling_width``` - (int) (Default: 2) The width of the pooling window.

```--pooling_height``` - (int) (Default: 2) The height of the pooling window.

```--hidden_layer_activation``` - (String) (Default: 'relu') The activation function of the hidden layers.

```--output_layer_activation``` - (String) (Default: 'softmax') The activation function of the output layer.