import argparse
import pandas as pd
import json
import psutil
import time
from cnvrg import Experiment

tic=time.time()
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--filename', action='store', dest='filename', default='/input/train_test_split/user1_movie_pred_whole.csv', required=True, help="""string. csv topics data file""")
parser.add_argument('--mapping_file_user', action='store', dest='mapping_file_user',default='/Matrix_Factorization/userdict.csv', required=True, help="""file which converts the users to integers""")
parser.add_argument('--mapping_file_item', action='store', dest='mapping_file_item',default='/Matrix_Factorization/itemdict.csv', required=True, help="""file which converts the users to integers""")
parser.add_argument('--top_count', action='store', dest='top_count',default=5, required=True, help="""file which converts the users to integers""")
parser.add_argument('--choice', action='store', dest='choice',default='/data/input/choice.csv', required=True, help="""file which contains the user choice of whose recommendation to get""")

parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")
parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")
args = parser.parse_args()
top_count=int(args.top_count)
fullpred1 = args.filename
mapping_user = args.mapping_file_user
mapping_item = args.mapping_file_item
fullpred = pd.read_csv(fullpred1)
mapping_item = pd.read_csv(mapping_item)
mapping_user = pd.read_csv(mapping_user)
choice = pd.read_csv(args.choice)
y=choice
tic=time.time()

#y=pd.DataFrame([0,10],columns=['user_id']) ###get recommendations for user 0 and 1
def predict(y):   
    # TODO: convert y to dataframe
    #y= pd.DataFrame({'user_id':y})
    mapping_user['originaluser_id'] = mapping_user['originaluser_id'].astype(str)
    mapping_item['originalitem_id'] = mapping_item['originalitem_id'].astype(str)
    ##### convert input user ids to strings and rename the userid column to originaluser_id
    user_id_input=pd.DataFrame(list(y['user_id'].astype('str')),columns=['originaluser_id'])
    ########convert user ids to internal ids
    converted_user_id=mapping_user.merge(user_id_input,on='originaluser_id',how='inner')
    #fullpred.dtypes
    ########get all the predictions for all items for the users requested
    results=fullpred.merge(converted_user_id,on='user_id',how='inner')
    print(results.columns)
    groups=results.groupby('user_id',sort=True)
    finaloutput=pd.DataFrame(columns=['originaluser_id','originalitem_id'])
    for group in groups.groups.keys():
        #######get top k recommendations for a user in the results
        recommendations=groups.get_group(group).sort_values('score',ascending=False).head(top_count)
        ###convert top k recommendations from internal to external
        recommendations=recommendations.merge(mapping_item,on='item_id',how='inner')[['originaluser_id','originalitem_id']]
        finaloutput=pd.concat([finaloutput, recommendations], ignore_index=True)
    finaloutput=finaloutput.rename({'originaluser_id':'user_id','originalitem_id':'item_id'},axis=1)
    return finaloutput

finaloutput_2 = predict(y)
finaloutput_2.to_csv("/cnvrg/{}".format("recommendations.csv" ), index=False)

