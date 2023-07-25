from __future__ import print_function
from surprise import NMF
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

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
import psutil
import time

tic=time.time()
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--train_file', action='store', dest='train_file', default='/data/movies_rec_sys/train_whole.csv', required=True, help="""training_file""")
parser.add_argument('--test_file', action='store', dest='test_file', default='/data/movies_rec_sys/test_whole.csv', required=True, help="""test_file""")
parser.add_argument('--num_of_steps_1', action='store', dest='num_of_steps_1', default=1, required=True, help="""number of iterations""")
parser.add_argument('--embed_dim_1', action='store', dest='embed_dim_1', default=50, required=True, help="""number of factors""")
parser.add_argument('--reg_pu', action='store', dest='reg_pu', default=0.06, required=True, help="""regularization coefficient""")
parser.add_argument('--reg_pi', action='store', dest='reg_pi', default=0.06, required=True, help="""REg coefficient """)
parser.add_argument('--threshold', action='store', dest='threshold', default=0.8, required=True, help="""threshold for choosing recommendations""")
parser.add_argument('--precision_at_value', action='store', dest='precision_at_value', default=10, required=True, help="""precision and recall at k""")

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file
n_iters_1 = int(args.num_of_steps_1)
n_factors_1 = int(args.embed_dim_1)
reg_pu = float(args.reg_pu)
reg_pi = float(args.reg_pi)
threshold = float(args.threshold)
K = int(args.precision_at_value)

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
max_v = ratings['rating'].max()
min_v = ratings['rating'].min()
nmf = NMF(n_factors=n_factors_1,verbose=False, n_epochs=n_iters_1,reg_pu=reg_pu, reg_qi=reg_pi,random_state=10)
reader = Reader(rating_scale=(min_v, max_v))
data = Dataset.load_from_df(train_whole[['user_id', 'item_id', 'rating']], reader) #ratings
trainset = data.build_full_trainset()

nmf.fit(trainset)
n_users = users['user_id'].unique().shape[0]
n_items = ratings['item_id'].unique().shape[0]
predic = np.zeros((n_users,n_items)) ##this is the output matrix #we need to declare n_users and n_items
for user in range(n_users):
    for item in range(n_items):
        predic[user , item] = nmf.predict(uid=user, iid=item).est

################## Computing Eval Metrics for SVG ########################
max_v = ratings['rating'].max()
user3_movie_pred_whole = pd.DataFrame(columns=['user_id','item_id','rating','score','error'])
eval_metrics_whole_nmf = pd.DataFrame(columns=['user_id','rmse','precision','recall'])
recommend_whole = pd.DataFrame(columns=['user_id','item_id','score'])

for nmf1 in range(len(users)):
  score_key_2 = 'score'
  scores_3 = pd.DataFrame({
      score_key_2: list(predic[nmf1]),
      'item_id': movies['item_id']
  })

  user3 = test_whole.loc[test_whole['user_id'] == nmf1]
  user3_movie = user3.merge(movies[['item_id']], on='item_id')
  user3_movie_pred = user3_movie.merge(scores_3[['item_id','score']])
  recommend = pd.DataFrame({
      'score': list(predic[nmf1]),
      'item_id': movies['item_id'],
      'user_id': nmf1
  })
  recommend = recommend[~recommend['item_id'].isin(train_whole.loc[(train_whole['user_id'] == nmf1)]['item_id'].to_list()+test_whole.loc[(test_whole['user_id'] == nmf1)]['item_id'].to_list()
)]
  recommend_whole = pd.concat([recommend_whole,recommend])  
  
  user_max2 = user3_movie_pred['score'].max()
  user3_movie_pred = user3_movie_pred.sort_values(by=['score'], ascending=False)
  relevant_cnt_nmf = user3_movie_pred.rating.loc[user3_movie_pred.rating >= threshold*max_v].count()
  recommended_cnt_nmf = user3_movie_pred.score.loc[user3_movie_pred.score >= threshold*user_max2].count()
  recommended_cnt_nmf_k = user3_movie_pred[:K].score.loc[user3_movie_pred.score >= threshold*user_max2].count()
  rec_rel_cnt_nmf = user3_movie_pred.rating.loc[(user3_movie_pred.rating >= threshold*max_v) & (user3_movie_pred.score >= threshold*user_max2)].count()
  rec_rel_cnt_nmf_k = user3_movie_pred[:K].rating.loc[(user3_movie_pred.rating >= threshold*max_v) & (user3_movie_pred.score >= threshold*user_max2)].count()
  precision_nmf = rec_rel_cnt_nmf/recommended_cnt_nmf
  precision_nmf_k = rec_rel_cnt_nmf_k/recommended_cnt_nmf_k
  recall_nmf = rec_rel_cnt_nmf/relevant_cnt_nmf
  recall_nmf_k = rec_rel_cnt_nmf_k/relevant_cnt_nmf
  user3_movie_pred['error'] = user3_movie_pred['score']-user3_movie_pred['rating']
  user3_movie_pred['error'] = user3_movie_pred['error']**2
  
  eval_metrics_nmf = pd.DataFrame(
      {'user_id' : nmf1,
       'rmse' : math.sqrt(user3_movie_pred['error'].mean()),
       'precision' : precision_nmf,
       'recall' : recall_nmf,
       'recall@k': recall_nmf_k,
       'precision@k':precision_nmf_k,
       'rel_count':relevant_cnt_nmf,
       'rec_count':recommended_cnt_nmf,
       'rel_rec_count':rec_rel_cnt_nmf,
       'rec_count_k':recommended_cnt_nmf_k,
       'rel_rec_count_k':rec_rel_cnt_nmf_k
       },index=[nmf1])
  eval_metrics_whole_nmf = pd.concat([eval_metrics_whole_nmf,eval_metrics_nmf])
  user3_movie_pred_whole = pd.concat([user3_movie_pred_whole, user3_movie_pred])

metrics_file_name = 'NMF_eval_metrics_file.csv'
dataset_pred_name = 'NMF_user1_movie_pred_whole.csv'
eval_metrics_file = 'NMF_eval_metrics_file.csv'
recommends_file = 'recommend.csv'

eval_metrics_whole_nmf.to_csv("/cnvrg/{}".format(metrics_file_name), index=False)
user3_movie_pred_whole.to_csv("/cnvrg/{}".format(dataset_pred_name), index=False)
recommend_whole.to_csv("/cnvrg/{}".format(recommends_file), index=False)

abc = eval_metrics_whole_nmf['precision'].mean()
abc1 = eval_metrics_whole_nmf['recall'].mean()
abc2 = eval_metrics_whole_nmf['rmse'].mean()
abc3 = eval_metrics_whole_nmf['precision@k'].mean()
abc4 = eval_metrics_whole_nmf['recall@k'].mean()
abc6 = eval_metrics_whole_nmf['rec_count_k'].mean()
abc7 = eval_metrics_whole_nmf['rel_rec_count_k'].mean()
abc8 = eval_metrics_whole_nmf['rel_count'].mean()
abc9 = eval_metrics_whole_nmf['rec_count'].mean()
abc10 = eval_metrics_whole_nmf['rel_rec_count'].mean()

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
e.log_param("NMF_ram", psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
e.log_param("NMF_time", toc-tic)