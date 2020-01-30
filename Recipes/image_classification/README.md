# Image Classification Flow Recipe
Image classification refers to a process in computer vision that can classify an image according to its visual content.  
For example, image classification model can decide whether a dog or cat in the image.

## Content
The attached yaml file describes a cnvrg flow build for image classification contains 3 AI libraries:  
1) ResNet50  
2) VGG16  
3) InceptionV3  
![unlinked](https://github.com/AccessibleAI/ailibrary/blob/master/Recipes/_docs/readme%20images/unliked_tasks.png)  

## Steps:  
(1) Add data-set task and connect it to the models tasks.  
![connect](https://github.com/AccessibleAI/ailibrary/blob/master/Recipes/_docs/readme%20images/data_linked_to_tasks.png)  

(2) Set the paths to the directories in the models tasks.  
![set_paths](https://github.com/AccessibleAI/ailibrary/blob/master/Recipes/_docs/readme%20images/set_data_path.png)  

(3) Repeat those steps for all tasks.  

(4) Push the play button.  




