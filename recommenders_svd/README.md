# SVD (Singular Value Decomposition), for Recommender System
_Calculates user and item embeddings, given the feedback matrix and minimizes the error between dot product of embedding matrices and the feedback matrix.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

SVD library is the third of the several algorithmic libraries in recommendation system blueprint for calculating the scores for the items that are not rated by the users.

## Features
- Upload the train, test files as well as enter the hyper parameters in the flow/experiment to start the library.
- A slightly different take than the matrix factorization, SVD's core features are its ability to decompose the reference matrix into 2 parts and accounting for user and item biases.
- It minimizes the error over unobserved values as well.

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
   - For best reuslts, the ratings must lie anywhere between 0 and 10. 
   - The file must be comma delimited and the headings must be in lower case.
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain float values as well as integer values.
- `--std_dev_1` is the standard deviation magnitude via which to asign the initial recommendations
- `--embed_dim_1`is the count of dimensions which will contain the embedding matrix. Higher dimensions mean more information stored but might overfit as well.
- `--num_of_steps_1` is the count of iterations, how many times the algorithm will run 
- `--learning_rate` is the rate at which the stochastic gradient descent will reach towards the end goal, i.e. make the product of embedding matrices U/V close to A, the reference matrix.
- `--reg_coef` is regularization coefficient, which penalizes the cost function further.
- `--threshold` is the`ratio, above which, both the predictions will be taken as having been recommended.
- `--rec_method`is the method of multiplication with which to multiply the matrices together. Could be DOT or COSINE.
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
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.cpu" --image=cnvrg:v5.0 --sync_before=false python3 SVD.py --train_file /input/train_test_split/train_whole.csv --test_file /input/train_test_split/test_whole.csv --num_of_steps_1 100 --embed_dim_1 30 --reg_coef 0.5 --learning_rate 0.5 --std_dev_1 0.5 --threshold 0.8 --precision_at_value 10
```

### About SVD
The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize. When baselines are not used, this is equivalent to Probabilistic Matrix Factorization
he prediction r^ui is set as:

### r^ui=μ+bu+bi+qTipu
If user u is unknown, then the bias bu and the factors pu are assumed to be zero. The same applies for item i with bi and qi.
To estimate all the unknown, we minimize the following regularized squared error:

### ∑rui∈Rtrain(rui−r^ui)2+λ(b2i+b2u+||qi||2+||pu||2)
The minimization is performed by a very straightforward stochastic gradient descent:

### bubipuqi←bu←bi←pu←qi+γ(eui−λbu)+γ(eui−λbi)+γ(eui⋅qi−λpu)+γ(eui⋅pu−λqi)
### where eui=rui−r^ui. 
These steps are performed over all the ratings of the trainset and repeated n_epochs times. Baselines are initialized to 0. User and item factors are randomly initialized according to a normal distribution, which can be tuned using the init_mean and init_std_dev parameters.

You also have control over the learning rate γ and the regularization term λ. Both can be different for each kind of parameter (see below). By default, learning rates are set to 0.005 and regularization terms are set to 0.02.
[Read More about it here](surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note)
[Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))