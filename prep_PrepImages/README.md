This library is made for pre-processing images.  

## Notes for this library
The library enables you to process images in order to build training and test sets easily.

## Parameters

```--path``` - (String) path to directory where the images are (required parameter).

```--height``` - (int) (Default: None) new height for resizing.

```--width``` - (int) (Default: None) new width for resizing.

```--grayscale``` - (bool) (Default: False) if the value is True, it turns all images to grayscale.

```--noise``` - (String) (Default: None) Represents types of noises can be added to images.
                 	Options are: (1) gaussian (2) s&p (3) speckle (4) poisson.

```--blur``` - (int) (Default: 0) Size of the squared kernel for gaussian blur.

```--convolve``` - (List) (Default: None) long list of lists of numbers which represents 2d squared array for the convolution to apply.

```--zip``` - (bool) (Default: False) if the value is True, it zips all the images to a zip named by the given directory name.

```--cnvrg_dataset_url``` - (String) (Default: None) cnvrg dataset url to push to created images to.


*** 07.04.20:  blurring and convolving doesn't work.
