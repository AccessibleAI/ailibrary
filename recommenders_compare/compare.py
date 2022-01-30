import pandas as pd
#import argparse
import os
import psutil
import time
from cnvrg import Experiment

tic=time.time()
#parser = argparse.ArgumentParser(description="""Preprocessor""")
#parser.add_argument('-f','--output_rec', action='store', dest='output_rec', default='/input/compare/recommend.csv', required=True, help="""training_file""")
#parser.add_argument('--test_file', action='store', dest='test_file', default='/data/movies_rec_sys/test_whole.csv', required=True, help="""test_file""")

#parser.add_argument('--project_dir', action='store', dest='project_dir',
#                        help="""--- For inner use of cnvrg.io ---""")
#parser.add_argument('--output_dir', action='store', dest='output_dir',
#                        help="""--- For inner use of cnvrg.io ---""")
#args = parser.parse_args()
for k in os.environ.keys():
    if 'PASSED_CONDITION' in k and os.environ[k] == 'true':
        print("Yes123")
        task_name = k.replace('CNVRG_', '').replace('_PASSED_CONDITION', '').lower()
#train_1 = args.output_rec
#train file = f'/input/{task_name}/{args.output_rec}'
train_file = '/input/'+task_name+'/'+'recommend.csv'
train = pd.read_csv(train_file)
file_name_variable = 'recommend.csv'
train.to_csv("/cnvrg/{}".format(file_name_variable), index=False)
print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
toc=time.time()
print("time taken:",toc-tic)
e = Experiment()
e.log_param("compare_ram", psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
e.log_param("compare_time", toc-tic)
#for i in os.environ:
#    print(i,'-',os.environ[i])
#    print("iteration")
