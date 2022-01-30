# NMF (Non-Negative Matrix Factorization), for Recommender System
_Calculates user and item embeddings, given the feedback matrix and minimizes the error between dot product of embedding matrices and the feedback matrix. It keeps the ratings as positive despite negative values (and thats how it differs from Matrix Factorization) _

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

NMF library is the sixth of the several algorithmic libraries in recommendation system blueprint for calculating the scores for the items that are not rated by the users.

## Features
- Upload the train, test files as well as enter the hyper parameters in the flow/experiment to start the library.
- A slightly different take than the normal Matrix factorization, it converts every value to a positive value and then proceeds with the factorization.)

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
   - For best reuslts, the ratings must lie within 0-10. However, implicit ratings (0,1) will work as well.
   - The file must be comma delimited and the headings must be in lower case.
   - The file path would contain the name of the dataset as well which contains the file. It would look like this :- `\dataset_name\input_file.csv`. Ensure that the dataset is of the text format.
   - The values can contain float values as well as integer values.
- `--embed_dim_1`is the count of dimensions which will contain the embedding matrix. Higher dimensions mean more information stored but might overfit as well.
- `--num_of_steps_1` is the count of iterations, how many times the algorithm will run 
- `--reg_pu` is regularization coefficient for items, which penalizes the cost function further.
-  `--reg_pu` is regularization coefficient for users, which penalizes the cost function further.
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
 - `--	user1_movie_pred_whole.csv` refers to the file which contains all the items existing in the test file along with their ratings and predicted scores. This file is used for manual verification.
    | user_no | item_no  | rating  | score | error
    | :---:   | :-: | :-: | :-: | :-:|
    | 0 | 1 | 3 | 2.7 | 0.09
    | 1 | 1 | 1 | 1.9 | 0.81
    | 1 | 2 | 2 | 0.5 | 2.25
- `-- eval_metrics_file.csv` refers to the file which contain the collection of all evaluation metrics, from precision and recall to the composite metric which is finally being used.
- user_id |	rmse | precision | recall | recall@k | precision@k | rel_count | rec_count | rel_rec_count | rec_count_k | rel_rec_count_k|
  | :---: | :-: | :-: | :-: | :-:| :---: | :-: | :-: | :-: | :-:| :-:|
  | 0 | 1.87 | 45% | 27% | 39%| 22% | 3.9 | 4.6 | 1.9 | 2.6| 1.7|

## How to run
```
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 NMF.py --train_file /input/train_test_split/train_whole.csv --test_file /input/train_test_split/train_whole.csv --num_of_steps_1 1000 --embed_dim_1 30 --reg_pu 0.06 --reg_pi 0.06 --threshold 0.8 --precision_at_value 10
```

### About NMF
A collaborative filtering algorithm based on Non-negative Matrix Factorization.
This algorithm is very similar to SVD. The prediction r^ui is set as:
##### r^ui=qTipu,
where user and item factors are kept positive.

The optimization procedure is a (regularized) stochastic gradient descent with a specific choice of step size that ensures non-negativity of factors, provided that their initial values are also positive.
At each step of the SGD procedure, the factors f or user u and item i are updated as follows:
##### pufqif←puf←qif⋅∑i∈Iuqif⋅rui∑i∈Iuqif⋅rui^+λu|Iu|puf⋅∑u∈Uipuf⋅rui∑u∈Uipuf⋅rui^+λi|Ui|qif
where λu and λi are regularization parameters.
This algorithm is highly dependent on initial values. User and item factors are uniformly initialized between init_low and init_high. Change them at your own risks!
A biased version is available by setting the biased parameter to True. In this case, the prediction is set as
##### r^ui=μ+bu+bi+qTipu,
still ensuring positive factors. Baselines are optimized in the same way as in the SVD algorithm. While yielding better accuracy, the biased version seems highly prone to overfitting so you may want to reduce the number of factors (or increase regularization).
[Read More about it here](surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note)
[Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))