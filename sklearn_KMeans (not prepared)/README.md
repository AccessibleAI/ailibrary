# K-Means

## General
**K-means Documentation**

*What is it?*

K-means is an unsupervised clustering technique dividing unsorted data into k distinct groups. This enables you to categorize new unseen based on what other items it is most similar to. 

*How does it work?*

The steps of the algorithm are the following.
1. Decide how many groups you want to be created. This number is k. 
2. The computer randomly selects k points to serve as the centroids, or focal point of each cluster. 
3. For each data point the machine computes which centroid is closest, and assigns the point to that centroid's group. 
4. For each cluster, the machine determines the average point of the whole cluster. These points will serve as the new centroids.
5. Repeat steps 3 and 4 until the centroids stop moving. Now 

*What are the advantages and disadvantages?*

Advantages: K-Means is simple to implement, scales to large data sets, guarantees convergence, can warm-start the positions of centroids, easily adapts to new examples, and generalizes to clusters of different shapes and sizes (e.g. elliptical clusters). 
Disadvantages: You must choose the initial k value manually; success is highly dependent on the initial centroid values; it is difficult to cluster data of varying size and density; unignored outlier points ruin the cluster structure; as the number of dimensions increases the distance poorly distinguishes various points.

## Note for this library


## Parameters
### cnvrg.io params
```--data``` - Required param. String. Path to .csv file (the dataset). Assumes that the files includes only integers and floats (no strings), and the table built like: all the columns but the 
rightmost one are considered as features columns (X), and the rightmost one is the label column (y).

```--test_split``` - Float. The portion of the data of testing. Default is 0.2.

```--output_model``` - String. The name of the output file which is a trained model. Default is kNearestNeighborsModel.sav

### algorithm params
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html