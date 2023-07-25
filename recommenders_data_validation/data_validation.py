import argparse
import pandas as pd
import psutil
import time
from cnvrg import Experiment

tic=time.time()
parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--filename', action='store', dest='filename', default='/data/movies_rec_sys/ratings_2.csv', required=True, help="""string. csv topics data file""")
parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")
parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")
args = parser.parse_args()
FILENAME = args.filename
df = pd.read_csv(FILENAME)
#if len(df['rating'].unique()) == 2:
#    df['rating'].replace(to_replace=1,value=2,inplace=True)
#    df['rating'].replace(to_replace=0,value=1,inplace=True)
#    print("Changed")
############## check column headings #############
headers=['user_id','item_id','rating']
if(all(df.columns==headers)==False):
        raise("Column headings not correct!")
#################### CHECK NAN #############
df=df.dropna()
#################### CHECK ratings are either integers or floats #############
try:
    df['rating']=df['rating'].astype('float')
except:
    print("Ratings have to be either integers or floats")
    raise()
########## Convert user and item ids to strings ##########

df['user_id']=df['user_id'].astype('str')

df['item_id']=df['item_id'].astype('str')

#################### CHECK ratings are between -10 and 10 #############

if(min(df['rating'])<-10 or max(df['rating'])>10):
    print("ratings have to be positive")
    raise()

##########normalize the ratings globally#########    
print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
    
#Create two dataframe mapping original user id and item id to internal representation and one dataframe of the original translated ratings frame
processed_dataframe=pd.DataFrame(columns=['user_id','item_id','rating'])

current_u_index = 0
current_i_index = 0

user = []
item = []
rating = []
raw2inner_id_users = {}
raw2inner_id_items = {}
# user raw id, item raw id, rating
for urid, irid, r in df.itertuples(index=False):
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1
            
            user.append(uid)
            item.append(iid)
            rating.append(r)
data={'originaluser_id':raw2inner_id_users.keys(),'user_id':raw2inner_id_users.values()}
convertuser=pd.DataFrame(data)
###########Total input size###########
print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))

print("number of users:",len(data))

data={'originalitem_id':raw2inner_id_items.keys(),'item_id':raw2inner_id_items.values()}
convertitem=pd.DataFrame(data)

print("number of items:",len(data))

data={'user_id':user,'item_id':item,'rating':rating}
processed_dataframe=pd.DataFrame(data) ####create a ready to use dataframe with converted values######    


full = "ratingstranslated.csv"
itemdict = "itemdict.csv" 
userdict = "userdict.csv" 
processed_dataframe.to_csv("/cnvrg/{}".format(full), index=False)
convertitem.to_csv("/cnvrg/{}".format(itemdict), index=False)
convertuser.to_csv("/cnvrg/{}".format(userdict), index=False)
convertitem.to_csv('/cnvrg/itemdict_1.csv')
convertuser.to_csv('/cnvrg/userdict_1.csv')

print('RAM GB used:', psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
toc=time.time()
print("time taken:",toc-tic)
e = Experiment()
e.log_param("dataval_ram", psutil.virtual_memory()[3]/(1024 * 1024 * 1024))
e.log_param("dataval_time", toc-tic)