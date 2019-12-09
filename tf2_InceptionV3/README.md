# InceptionV3

## General


## Note for this library
The library is a wrapper for the ```Tensorflow2``` implementation of the InceptionV3 module. The library enables the user to create deep neural network based on InceptionV3.
It might be used both for colored ('rgb') images and for black-white image ('grayscale') by the parameter ```--image_color```.
The library also enables the user to load the images directly from the local directory. By default, the library needs the recive single path (``--data``) for a local directory.
When doing this, the library splits the content of the directory to train set and validation/test set according to the parameter ```--test_size``` which by default equals 0.2.
If the user is also interested in supplying test set, it can be given to the parameter ```--data_test```. If the user supplied it, the library performs: training, validation and testing.

## Parameters
```--data``` - (String) (Required param) Path to a local directory which contains sub-directories, each for a single class. The data is used for training and validation.

```--data_test``` - (String) (Default: None) Path to a local directory which contains sub-directories, each for a single class. The data is used for testing. 

```--output_model``` - (String) (Default: 'model.h5') The name of the output model file. It is recommended to use '.h5' file.

```test_size``` - (float) (Default: 0.2) The size of the validation / test set. If test set supplied, it represents the size of the validation set out of the data 
set given in --data. Otherwise, it represents the size of the test set out of the data set given in --data.

```epochs``` - (int) (Default: 1) The number of epochs the algorithm performs in the training phase.

```batch_size``` - (int) (Default: 256) The number of images the generator downloads in each step.

```image_color``` - (String) (Default: 'rgb') The colors of the images. Can be one of: 'grayscale', 'rgb'.

```--optimizer``` - (String) (Default: 'adam') The optimizer the algorithm uses. Can be one of: 'adam', 'adagrad', 'rmsprop', 'sgd'.

```image_height``` - (int) (Default: 200) The height of the images.

```image_width``` - (int) (Default: 200) The width of the images.

```conv_width``` - (int) (Default: 3) The width of the convolution window.

```conv_height``` - (int) (Default: 3) The height of the convolution window.

```pool_width``` - (int) (Default: 2) The width of the pooling window.

```pool_height``` - (int) (Default: 2) The height of the pooling window.

```hidden_layer_activation``` - (String) (Default: 'relu') The activation function of the hidden layers.

```output_layer_activation``` - (String) (Default: 'softmax') The activation function of the output layer.