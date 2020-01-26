# The cnvrg AI Library
The AI Library is an ML package manager.  
Build, share and reuse machine learning components that are packaged with code, docker containers and compute engine types.  
Quickly execute across any kind of infrastructure and automatically track hyper-parameters, metrics, utilization, and more.  

## Content
1) **'sk'** directories: Those include the AI libraries for machine learning which are backed by scikit learn.
Scikit learn is a free software machine learning module in Python language. It features various classification, regression and clustering algorithms.
In the cnvrg library we use mostly classification, regression and clustering algorithms.

2) **'tf2'** directories: Those include the AI libraries for deep learning. The purpose of those
libraries is not to build or create models, but rather to leverage prepared model structures.  
Currently, the library includes commonly known successful models structures like ResNet50 and vgg16, and they are imported
with ImageNet weights, so when used they perform transfer learning.

3) **'pr'** directories: Those include libraries which enables pre-processing of various types of files.

3) Directories begins with **'_'**: Those are not for user's use, but for internal testing at cnvrg.io and version control.

## Metrics
There are some metrics which are created in the libraries.  
The metrics are changed between families of algorithm (e.g: classification and regression algorithms has different metrics).  

### Classification Metrics
```model``` - The name of the output model file.  
```folds``` - The number of folds used in the cross validation (if demanded by the user).  
```train_acc``` - float in (0, 1).  
```train_loss``` - calculated by default by mean_squared_error.  
```test_acc``` - float in (0, 1).  
```test_loss``` - calculated by default by mean_squared_error.  
Example:  


### Regression Metrics
```model``` - The name of the output model file.  
```folds``` - The number of folds used in the cross validation (if demanded by the user).  
```train_loss_MSE``` - The train error evaluated by mean squared error.  
```train_loss_MAE``` - The train error evaluated by mean absolute error.  
```train_loss_R2``` - The train error evaluated by R2.  
```test_loss_MSE``` - The test error evaluated by mean squared error.  
```test_loss_MAE``` - The test error evaluated by mean absolute error.  
```test_loss_R2``` - The test error evaluated by R2.  
Example:  


### Deep Learning Metrics
**SOON**


### Clustering Metrics
**SOON**  
Example: 


## Visualizations
There are several types of visualizations which are built-in the libraries and are plotted automatically.  
Not all the libraries plot all the possible visualizations due to differences in the the algorithms.  

#### Classification Report
Classification Report is a textual report shows some main classification metrics.  
It is important to notice that classification report is relevant only for classification algorithms.  
The metrics which it shows are: precision, recall, f1-score and support.  
* Precision - 

* Recall - 

* F1-score -

* Support -

Examples:
1) Iris data set (3 labels):  
![first_class_rep](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/classification_report.png)  

2) 2-labels data set:  
![second_class_rep](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/classification_report_2.png)

#### Confusion Matrix
Confusion Matrix is a table which evaluates accuracy by computing the confusion matrix with each row corresponding to the true class.  
Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa).  
The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another).  
For instance, lets say we have a data set with 4 different labels: a, b, c, d.  
So, the confusion matrix is going to be 4 * 4 table where the y-axis describes the true label, and the x-axis describes the predicted label.  

Examples:  
1) 2-labels data set:  
![first_conf_mat](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/confustion_matrix.png)

2) 3-labels data set:  
![second_conf_mat](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/confustion_matrix_2.png)

#### Feature Importance
Feature importance is a bars chars where the features names are the x-axis and the y-axis is a range of floats between 0 to 1 (or the opposite).  
Each bar, belongs the single feature, represents a float number which is the score of importance of the specific feature in the prediction process.  
It can be used for feature selection, dimensionality reduction, improving estimators accuracy and boosting performance.  

Examples:  
1)  
![first_fea_imp](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/feature_importance.png)

2)  
![second_fea_imp](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/feature_importance_3.png)

#### ROC curve
A ROC (receiver operating characteristic curve) is a graphical plot that illustrates the diagnostic ability of a binary classifier.  
The ROC curve is created by plotting the true positive rate (TP/(TP+FN)) against the false positive rate (FP/(FP + TN)) at various threshold settings.  
Note: this implementation is restricted to the binary classification task.  

Examples:  
1)  
![first_roc_curve](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/roc_curve_1.png)  

#### Correlation
Correlation is any statistical relationship between two random variables or bivariate data.  
In the broadest sense correlation is any statistical association, though it commonly refers to the degree to which a pair of variables are linearly related.  
Correlations are useful because they can indicate a predictive relationship that can be exploited in practice.  
Data Scientist or Statisticians might be interested in correlation in order to make dimensionality reduction, improving estimators accuracy and boosting performance.  

Examples:
1)  
![first_corr](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/correlation.png)  

#### feature-vs-feature
When our library detects strong correlation (smaller than -0.7 or greater than 0.7), it automatically produces a scatter plot which presents the correlation.  

Examples:  
1) Watch the following correlation table:  
![second_corr](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/correlation_2.png)  
As you can see (in the popped window), there is a high correlation (0.93) between the two features.  
So, the library automatically produces the following scatter plot:  
![first-fea-vs-fea](https://github.com/AccessibleAI/ailibrary/blob/master/_docs/readme_images/feature_against_feature_2.png)  
Which shows the strong relation between the two (where the first grows, the second grows almost linearly).  
