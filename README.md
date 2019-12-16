# Welcome to cnvrg's AI Library

## What is AI Library?

The AI Library is an ML package manager. Build, share and reuse machine learning components that are packaged with code, docker container and compute engine type. Run quickly across any kind of infrastructure and automatically track hyperparameters, metrics, usage, and more.


## Content

1) 'sk' directories - Those include the AI libraries for machine learning which are backed by scikit learn.
Scikit learn is a free software machine learning library in Python language. It features various classification, regression and clustering algorithms.
In the cnvrg.io library we use mostly classification algorithms and some for regression.

2) 'tf2' directories - Those include the AI libraries for deep learning. Unlike tensorflow, the purpose of those
libraries is not to allow or support building and creating models, but use prepared models structures in different uses.
Currently, the library includes commonly known successful models structures like ResNet50 and vgg16, and they are imported
with ImageNet weights on them, so the user actually performs transfer learning in his training process.

3) directories begins with '_' - Those are not for user's use, but for inner testing at cnvrg.io and version controlling.