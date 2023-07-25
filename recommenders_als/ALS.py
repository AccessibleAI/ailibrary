from __future__ import print_function
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow.compat.v1 as tf
import argparse
import math
from cnvrg import Experiment
tf.disable_v2_behavior()
import psutil
import time

tic=time.time()
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--train_file', action='store', dest='train_file', default='/data/movies_rec_sys/train_whole.csv', required=True, help="""training_file""")
parser.add_argument('--test_file', action='store', dest='test_file', default='/data/movies_rec_sys/test_whole.csv', required=True, help="""test_file""")
parser.add_argument('--num_of_steps_1', action='store', dest='num_of_steps_1', default=100, required=True, help="""number of iterations""")
parser.add_argument('--embed_dim_1', action='store', dest='embed_dim_1', default=50, required=True, help="""number of factors""")
parser.add_argument('--reg_coef', action='store', dest='reg_coef', default=0.02, required=True, help="""regularization coefficient""")
parser.add_argument('--threshold', action='store', dest='threshold', default=0.8, required=True, help="""threshold for choosing recommendations""")
parser.add_argument('--precision_at_value', action='store', dest='precision_at_value', default=10, required=True, help="""precision and recall at k""")

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file
n_iters_1 = int(args.num_of_steps_1)
n_factors_1 = int(args.embed_dim_1)
reg_coef_1 = float(args.reg_coef)
threshold = float(args.threshold)
K = int(args.precision_at_value)

hyp = pd.DataFrame(columns=['dimension','reg_coef'])
hyp.at[0,'dimension'] = n_factors_1
hyp.at[0,'reg_coef'] = reg_coef_1
hyp.to_csv('hyp.csv')
hyp_file = 'hyp1.csv'
hyp.to_csv("/cnvrg/{}".format(hyp_file), index=False)

train_whole = pd.read_csv(train_file)
test_whole = pd.read_csv(test_file)
train_whole['user_id'] = train_whole['user_id'].astype(int)
train_whole['item_id'] = train_whole['item_id'].astype(int)
test_whole['user_id'] = test_whole['user_id'].astype(int)
test_whole['item_id'] = test_whole['item_id'].astype(int)

ratings = pd.concat([train_whole,test_whole])
users = ratings['user_id'].to_frame().drop_duplicates().sort_values('user_id').reset_index().drop('index',1)
ratings['user_id'] = ratings['user_id'].astype(int)
ratings['item_id'] = ratings['item_id'].astype(int)
users['user_id'] = users['user_id'].astype(int)

movies = pd.DataFrame({'item_id' : ratings['item_id'].unique()})

df=ratings #the original ratings dataframe
n_users = df['user_id'].unique().shape[0]
n_items = df['item_id'].unique().shape[0]
ratings = np.zeros((n_users, n_items))

for row in df.itertuples(index = False):
    ratings[int(row.user_id) , int(row.item_id) ] = row.rating

def create_train_test(n_users,n_items,tr,ts):
    train = np.zeros((n_users, n_items))
    test = np.zeros((n_users, n_items))
    for row in tr.itertuples(index = False):
            train[int(row.user_id) , int(row.item_id) ] = row.rating
    for row in ts.itertuples(index = False):
            test[int(row.user_id) , int(row.item_id) ] = row.rating
    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)
    return train, test

train, test = create_train_test(n_users,n_items,train_whole,test_whole) #traino is the training dataframe and 

class ExplicitMF:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank
        
    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """
    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        
    def fit(self, train, test):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        # self.user_factors = np.random.random((self.n_user, self.n_factors))
        # self.user_factors = np.random.normal((self.n_user, self.n_factors))
        # self.item_factors = np.random.random((self.n_item, self.n_factors))
       # print(self.n_item)
       # print(self.n_factors)
        np.random.seed(10)
        self.user_factors = np.random.normal(0,0.05,(self.n_user, self.n_factors))
       # print(self.user_factors)
        self.item_factors = np.random.normal(0,0.05,(self.n_item, self.n_factors))
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.test_mse_record  = []
        self.train_mse_record = []   
        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors) 
            predictions = self.predict()
            test_mse = self.compute_mse(test, predictions)
            train_mse = self.compute_mse(train, predictions)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)
        
        return self    
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse

als = ExplicitMF(n_iters = n_iters_1, n_factors = n_factors_1, reg = reg_coef_1)
als.fit(train, test)
alsoutput=als.predict()
max_v = df['rating'].max()
user2_movie_pred_whole = pd.DataFrame(columns=['user_id','item_id','rating','score','error'])
eval_metrics_whole_als = pd.DataFrame(columns=['user_id','rmse','precision','recall'])
recommend_whole = pd.DataFrame(columns=['user_id','item_id','score'])

for als1 in range(len(users)):
  #scores = compute_scores(model.embeddings["user_id"][il], model.embeddings["item_id"], measure=DOT)
  score_key_1 = 'score'
  scores_2 = pd.DataFrame({
      score_key_1: list(alsoutput[als1]),
      'item_id': movies['item_id']
  })
  user2 = test_whole.loc[test_whole['user_id'] == als1]

  user2_movie = user2.merge(movies[['item_id']], on='item_id')
  user2_movie_pred = user2_movie.merge(scores_2[['item_id','score']])

  recommend = pd.DataFrame({
      'score': list(alsoutput[als1]),
      'item_id': movies['item_id'],
      'user_id': als1
  })
  recommend = recommend[~recommend['item_id'].isin(train_whole.loc[(train_whole['user_id'] == als1)]['item_id'].to_list()+test_whole.loc[(test_whole['user_id'] == als1)]['item_id'].to_list()
)]
  recommend_whole = pd.concat([recommend_whole,recommend])  
    
  user_max1 = user2_movie_pred['score'].max()
  user2_movie_pred = user2_movie_pred.sort_values(by=['score'], ascending=False)
  relevant_cnt_als = user2_movie_pred.rating.loc[user2_movie_pred.rating >= threshold*max_v].count()
  recommended_cnt_als = user2_movie_pred.score.loc[user2_movie_pred.score >= threshold*user_max1].count()
  recommended_cnt_als_k = user2_movie_pred[:K].score.loc[user2_movie_pred.score >= threshold*user_max1].count()
  rec_rel_cnt_als = user2_movie_pred.rating.loc[(user2_movie_pred.rating >= threshold*max_v) & (user2_movie_pred.score >= threshold*user_max1)].count()
  rec_rel_cnt_als_k = user2_movie_pred[:K].rating.loc[(user2_movie_pred.rating >= threshold*max_v) & (user2_movie_pred.score >= threshold*user_max1)].count()
  precision_als = rec_rel_cnt_als/recommended_cnt_als
  recall_als = rec_rel_cnt_als/relevant_cnt_als
  precision_als_k = rec_rel_cnt_als_k/recommended_cnt_als_k
  recall_als_k = rec_rel_cnt_als_k/relevant_cnt_als
  user2_movie_pred['error'] = user2_movie_pred['score']-user2_movie_pred['rating']
  user2_movie_pred['error'] = user2_movie_pred['error']**2
  
  eval_metrics_als = pd.DataFrame(
      {'user_id' : als1,
       'rmse' : math.sqrt(user2_movie_pred['error'].mean()),
       'precision' : precision_als,
       'recall' : recall_als,
       'recall@k': recall_als_k,
       'precision@k':precision_als_k,
       'rel_count':relevant_cnt_als,
       'rec_count':recommended_cnt_als,
       'rel_rec_count':rec_rel_cnt_als,
       'rec_count_k':recommended_cnt_als_k,
       'rel_rec_count_k':rec_rel_cnt_als_k
       },index=[als1])
  eval_metrics_whole_als = pd.concat([eval_metrics_whole_als,eval_metrics_als])
  user2_movie_pred_whole = pd.concat([user2_movie_pred_whole, user2_movie_pred])

metrics_file_name = 'eval_metrics_file_als.csv'
dataset_pred_name = 'user2_movie_pred_whole.csv'
recommends_file = 'recommend.csv'

eval_metrics_whole_als.to_csv("/cnvrg/{}".format(metrics_file_name), index=False)
user2_movie_pred_whole.to_csv("/cnvrg/{}".format(dataset_pred_name), index=False)
recommend_whole.to_csv("/cnvrg/{}".format(recommends_file), index=False)

abc = eval_metrics_whole_als['precision'].mean()
abc1 = eval_metrics_whole_als['recall'].mean()
abc2 = eval_metrics_whole_als['rmse'].mean()
abc3 = eval_metrics_whole_als['precision@k'].mean()
abc4 = eval_metrics_whole_als['recall@k'].mean()
abc6 = eval_metrics_whole_als['rec_count_k'].mean()
abc7 = eval_metrics_whole_als['rel_rec_count_k'].mean()
abc8 = eval_metrics_whole_als['rel_count'].mean()
abc9 = eval_metrics_whole_als['rec_count'].mean()
abc10 = eval_metrics_whole_als['rel_rec_count'].mean()

e = Experiment()
e.log_param("precision", abc)
e.log_param("recall",abc1)
e.log_param("rmse",abc2)
e.log_param("precision_at_k",abc3)
e.log_param("recall_at_k",abc4)
e.log_param("relevant_cnt",abc8)
e.log_param("recommended_cnt",abc9)
e.log_param("rel_rec_cnt",abc10)
e.log_param("recommended_cnt_k",abc6)
e.log_param("rel_rec_count_k",abc7)
composite_metric = (abc10*(1/100)*0.1) + (1/abc2)*(0.4) + (abc4*0.25) + (abc3*0.25)
e.log_param("compositemetric",composite_metric)
print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
toc=time.time()
print("time taken:",toc-tic)
e.log_param("ALS_ram", psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
e.log_param("ALS_time", toc-tic)