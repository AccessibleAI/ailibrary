import os
import argparse
import requests
import pandas as pd
import datetime
import csv
import dateutil



class DaysAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 0 < values < 7:
            raise argparse.ArgumentError(self, "port numbers must be between 1 and 6")
        setattr(namespace, self.dest, values)


class EndPointAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values == 'recent' and not values == 'all':
            raise argparse.ArgumentError(self, "endpoint must be 'recent'  or 'all' ")
        setattr(namespace, self.dest, values)


parser = argparse.ArgumentParser(description="""Preprocessor""")

parser.add_argument('--token', action='store', dest='token', required=True,
					help="""string. bearer token for tweeter access.""")


parser.add_argument('--term', action='store', dest='term', required=True,
					help="""string. keyword for twetter search.""")

parser.add_argument('-d','--dataset', action='store', dest='dataset', required=False,
					help="""string. dataset for saving the result.""")


parser.add_argument('-o','--output_file', action='store', dest='output_file' ,default='twitts.csv' ,required=False,
					help="""string. filename for saving the data""")                    


parser.add_argument('-m','--max_twitts', action='store', dest='max_twitts' ,default=500 ,required=False,
					type=int, help="""max num of twitts""")  


parser.add_argument('-e','--end_point', action=EndPointAction, dest='endpoint' ,default='recent' ,required=False,
					help="""string. twitter api endpoint""") 


parser.add_argument('-b','--days_back', action=DaysAction, dest='days_back' ,default=3 ,required=False,
					type=int, help="""num of days back for recent twitts""") 



args = parser.parse_args()
token = args.token
term = args.term
dataset = args.dataset
max_twitts = args.max_twitts
endpoint = args.endpoint
days_back = args.days_back
OUTFILE = '/cnvrg/{}'.format(args.output_file)




print("token = ",token)
print("search term = ",term)
print("output file = ", OUTFILE)
print("max twitts  = ", max_twitts)
print("endpoint = ", endpoint)
print("days back = ", days_back)
print("dir = ",os.getcwd())






# Create file
csvFile = open(OUTFILE, "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)

#Create columns 
csvWriter.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','text'])
csvFile.close()


def append_to_csv(json_response, fileName):

    counter = 0

    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    for tweet in json_response['data']:
        
        author_id = tweet['author_id']
        created_at = dateutil.parser.parse(tweet['created_at'])
 
        if ('geo' in tweet):   
            geo = tweet['geo']['place_id']
        else:
            geo = " "

        tweet_id = tweet['id']
        lang = tweet['lang']

        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        source = tweet['source']
        text = tweet['text']
        
        res = [author_id, created_at, geo, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source, text]
        
        csvWriter.writerow(res)
        counter += 1

    csvFile.close()
    print("# of Tweets added from this response: ", counter) 




def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/{}".format(endpoint) 

    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()



def get_last_tweets(token, term):
    #Inputs for the request
    flag = True
    next_token = None
    total_data = 0

    bearer_token = token
    headers = create_headers(bearer_token)
    keyword = term + " lang:en"
    d = datetime.timedelta(minutes = 5)
    now =  n = datetime.datetime.now(datetime.timezone.utc)
    now = now - d
    d = datetime.timedelta(days = days_back)
    ago = now - d 
    start_time = ago.isoformat().split('+')[0]+'Z'
    end_time = now.isoformat().split('+')[0]+'Z'
    max_results = 100
    url = create_url(keyword, start_time,end_time, max_results)
    while flag == True:
        json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
        append_to_csv(json_response, OUTFILE)
        next_token = json_response['meta']['next_token']
        data_len = len(json_response['data'])
        total_data += data_len
        print('len of data = ',data_len)
        if next_token == None or total_data >= max_twitts:
            flag = False
    if dataset is not None:
        if not os.path.exists(dataset): #Check if the folder exits
            os.mkdir(dataset) #If it doesn't, create it
        os.system("cd {} && cnvrg data init && cnvrg data put ../{}".format(dataset, OUTFILE))


get_last_tweets(token,term)


