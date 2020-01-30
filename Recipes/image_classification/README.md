# Image Classification Flow Recipe
Image classification refers to a process in computer vision that can classify an image according to its visual content.  
For example, image classification model can decide whether a dog or cat in the image.

The attached yaml file describes a cnvrg flow build for image classification contains 3 AI libraries:  
1) ResNet50  
2) VGG16  
3) InceptionV3  
And then: 
In order to maximize the of the following flow, the user should attach data-set task and connect it to the models tasks.
After connecting the tasks visually, the user should set the paths to the directories in the models tasks.  
