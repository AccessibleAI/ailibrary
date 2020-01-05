# Welcome to cnvrg's AI Library

## What is AI Library?

The AI Library is an ML package manager.\
Build, share and reuse machine learning components that are packaged with code, docker container and compute engine type.\
Run quickly across any kind of infrastructure and automatically track hyperparameters, metrics, usage, and more.


Content
-

1) 'sk' directories - Those include the AI libraries for machine learning which are backed by scikit learn.
Scikit learn is a free software machine learning library in Python language. It features various classification, regression and clustering algorithms.
In the cnvrg.io library we use mostly classification algorithms and some for regression.

2) 'tf2' directories - Those include the AI libraries for deep learning. Unlike tensorflow, the purpose of those
libraries is not to allow or support building and creating models, but use prepared models structures in different uses.
Currently, the library includes commonly known successful models structures like ResNet50 and vgg16, and they are imported
with ImageNet weights on them, so the user actually performs transfer learning in his training process.

3) directories begins with '_' - Those are not for user's use, but for inner testing at cnvrg.io and version controlling.


Visualizations
-

There are several types of visualizations which are built-in the libraries and are plotted automatically. \
Not all the libraries plot all the possible visualizations due to differences in the the algorithms.\

### Visualization Types:

#### Classification Report
Classification Report is a textual report shows some main classification metrics.\
It is important to notice that classification report is relevant only for classification algorithms.\
The metrics which it shows are: precision, recall, f1-score and support.
* Precision -

* Recall - 

* F1-score -

* Support -

Examples:

![first](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/classification_report.png)


#### Confusion Matrix
Confusion Matrix is a table which evaluates accuracy by computing the confusion matrix with each row corresponding to the true class.\
Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).\
The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another).\
For instance, lets say we have a data set with 4 different labels: a, b, c, d.\
So, the confusion matrix is going to be 4 * 4 table where the y-axis describes the true label, and the x-axis describes the predicted label.

Examples:


#### Feature Importance
Feature importance is a bars chars where the features names are the x-axis and the y-axis is a range of floats between 0 to 1 (or the opposite).\
Each bar, belongs the single feature, represents a float number which is the score of importance of the specific feature in the prediction process.\
It can be used for feature selection, dimensionality reduction, improving estimators accuracy and boosting performance.

Examples:
 

