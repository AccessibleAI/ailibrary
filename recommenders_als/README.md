# ALS (Alternating Least Squares), for Recommender System
_Calculates user and item embeddings, given the feedback matrix and minimizes the error between dot product of embedding matrices and the feedback matrix. ALS is another approach to minimizing the error. All other algorithms in the recommendation system blueprint minimize error using Stochastic Gradient Descent.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

ALS library is the fifth of the several algorithmic libraries in recommendation system blueprint for calculating the scores for the items that are not rated by the users.

## Features
- Upload the train, test files as well as enter the hyper parameters in the flow/experiment to start the library.
- The cost function taken is similar to the SVD algorithm, however the main feature is the method employed to minimize the cost function, which is called Alternating Least Squares.
- It minimizes the error over observed values.

## Arguments
- `--train_file` refers to the name and path of the training file which either can come from train test split or from a custom upload..
    | user_no | item_no  | rating  |
    | :---:   | :-: | :-: |
    | 0 | 3 | 4 |
    | 1 | 1 | 1 |
    | 1 | 2 | 1 |
- `--test file` refers to the name and path of the training file which either can come from train test split or from a custom upload..
    | user_no | item_no  | rating  |
    | :---:   | :-: | :-: |
    | 0 | 3 | 4 |
    | 1 | 1 | 1 |
    | 1 | 2 | 1 |
- Both need to satisfy some criteria
   - The user_no and item_no must start at 0. 
   - For best results, the range of ratings must lie anywhere between 0 and 10. 
   - The file must be comma delimited and the headings must be in lower case.
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain float values as well as integer values.
- `--embed_dim_1`is the count of dimensions which will contain the embedding matrix. Higher dimensions mean more information stored but might overfit as well.
- `--num_of_steps_1` is the count of iterations, how many times the algorithm will run 
- `--reg_coef` is regularization coefficient, which penalizes the cost function further.
- `--threshold` is the rating value, above which, both the predictions will be taken as having been recommended.
- `--precision_at_value`gives users the ability to customize the number at which they want to see the precision/recall. If this value is 5, then users will see precision and recall at the top 5 choices, sorted.

## Model Artifacts 
- `--recommend.csv` refers to the file which contains items not rated by the user, making them ideal for final recommendation. This file can be rather large, depending on the number of distinct items and users in the original file.
    | user_no | item_no  | score  |
    | :---:   | :-: | :-: |
    | 0 | 5 | 2.6 |
    | 1 | 3 | 1.7 |
    | 1 | 6 | 4.4 |
 - `--  user1_movie_pred_whole.csv` refers to the file which contains all the items existing in the test file along with their ratings and predicted scores. This file is used for manual verification.
    | user_no | item_no  | rating  | score | error
    | :---:   | :-: | :-: | :-: | :-:|
    | 0 | 1 | 3 | 2.7 | 0.09
    | 1 | 1 | 1 | 1.9 | 0.81
    | 1 | 2 | 2 | 0.5 | 2.25
- `-- eval_metrics_file.csv` refers to the file which contain the collection of all evaluation metrics, from precision and recall to the composite metric which is finally being used.
- user_id | rmse | precision | recall | recall@k | precision@k | rel_count | rec_count | rel_rec_count | rec_count_k | rel_rec_count_k|
  | :---: | :-: | :-: | :-: | :-:| :---: | :-: | :-: | :-: | :-:| :-:|
  | 0 | 1.87 | 45% | 27% | 39%| 22% | 3.9 | 4.6 | 1.9 | 2.6| 1.7|

## How to run
```
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.cpu" --image=cnvrg:v5.0 --sync_before=false python3 SVD.py --train_file /input/train_test_split/train_whole.csv --test_file /input/train_test_split/test_whole.csv --num_of_steps_1 100 --embed_dim_1 30 --reg_coef 0.5 --threshold 0.8 --precision_at_value 10
```

### About ALS
We start by denoting our **d** feature user into math by letting a user **u** take the form of a **1×d**-dimensional vector **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;x_{u})**. These for often times referred to as latent vectors or low-dimensional embeddings. Similarly, an item **i** can be represented by a **1×d**-dimensional vector **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;y_{i})**. And the rating that we predict user **u** will give for item **i** is just the dot product of the two vectors

![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;\hat{r}_{ui}=x_{u}y_{i}^{T}=\sum_{d}^{}x__{ud}y_{di})
Where ![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;\hat{r}__{ui}) represents our prediction for the true rating ![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;\{r}__{ui}). Next, we will choose a objective function to minimize the square of the difference between all ratings in our dataset (**S**) and our predictions. This produces a objective function of the form:

# L=∑u,i∈S(rui−xuyTi)2+λ(∑u∥xu∥2+∑i∥yi∥2)

Note that we've added on two **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;\{L}__{2})** regularization terms, with **λ** controlling the strength at the end to prevent overfitting of the user and item vectors. **λ**, is another hyperparameter that we'll have to search for to determine the best value. The concept of regularization can be a topic of itself, and if you're confused by this, consider checking out this documentation that covers it a bit more.

Now that we formalize our objective function, we'll introduce the **Alternating Least Squares (ALS)** method for optimizing it. The way it works is we start by treating one set of latent vectors as constant. For this example, we'll pick the item vectors, **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;y_{i})**. We then take the derivative of the loss function with respect to the other set of vectors, the user vectors, **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;x_{u})** and solve for the non-constant vectors (the user vectors).

# ∂L∂xu=−2∑i(rui−xuyTi)yi+2λxu=0=−(ru−xuYT)Y+λxu=0=xu(YTY+λI)=ruY=xu=ruY(YTY+λI)−1

To clarify it a bit, let us assume that we have **m** users and **n** items, so our ratings matrix is **mxn**.

The row vector  **![r_{ui}=μ](https://latex.codecogs.com/svg.latex?\Large&space;r_{u})** represents users u's row from the ratings matrix with all the ratings for all the items (so it has dimension **1×n**)
We introduce the symbol **Y**, with dimensions **n×d**, to represent all item row vectors vertically stacked on each other
Lastly, **I** is the identity matrix which has dimension **d×d** to ensure the matrix multiplication's dimensionality will be correct when we add the **λ**
Now comes the alternating part: With these newly updated user vectors in hand, in the next round, we hold them as constant, and take the derivative of the loss function with respect to the previously constant vectors (the item vectors). As the derivation for the item vectors is quite similar, we will simply list out the end formula:

# ∂L∂yi=yi=riX(XTX+λI)−1

Then we alternate back and forth and carry out this two-step process until convergence. The reason we alternate is, optimizing user latent vectors, **U**, and item latent vectors **V** simultaneously is hard to solve. If we fix **U** or **V** and tackle one problem at a time, we potentially turn it into a easier sub-problem. Just to summarize it, ALS works by:

1. Initialize the user latent vectors, **U**, and item latent vectors V randomly
2. Fix  **U** and solve for **V**
3. Fix **V** and solve for **U**
4. Repeat step 2 and 3 until convergence. 

# Reference
[Read More about it here](http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html#Reference)