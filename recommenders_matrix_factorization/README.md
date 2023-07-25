# Basic Matrix Factorization, for Recommender System
_Calculates user and item embeddings, given the feedback matrix and minimizes the error between dot product of embedding matrices and the feedback matrix _

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Matrix Factorization library is the first of the several algorithmic libraries in recommendation system blueprint for calculating the scores for the items that are not rated by the users.

## Features
- Upload the train, test files as well as enter the hyper parameters in the flow/experiment to start the library.
- Simpest method for factorization, tends to give a moderate magnitude of score resulting in average precision/recall values
- Tends to be sometimes biased towards popular choices. 

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
- `--std_dev_1` is the standard deviation magnitude via which to asign the initial recommendations
- `--embed_dim_1`is the count of dimensions which will contain the embedding matrix. Higher dimensions mean more information stored but might overfit as well.
- `--num_of_steps_1` is the count of iterations, how many times the algorithm will run 
- `--learning_rate` is the rate at which the stochastic gradient descent will reach towards the end goal, i.e. make the product of embedding matrices U/V close to A, the reference matrix.
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
cnvrg run  --datasets='[{id:"movies_rec_sys",commit:"f6e1126cedebf23e1463aee73f9df08783640400"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Matrix_Factorization.py --train_file /input/train_test_split/train_whole.csv --test_file /input/train_test_split/test_whole.csv --std_dev_1 0.3 --embed_dim_1 30 --num_of_steps_1 1000 --learning_rate 0.5 --threshold 0.8 --rec_method DOT --precision_at_value 10
```

### About Matrix Factorization
Matrix factorization is a simple embedding model. Given the feedback matrix A [R^m*n] ,where m is the number of users (or queries) and n is the number of items, the model learns
A user embedding matrix U [R^m*d], where row i is the embedding for user i.
An item embedding matrix V [R^n*d] , where row j is the embedding for item j.
The embeddings aer such that the dot product of U & V is a good approximation of A.
[Read More about it here](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)