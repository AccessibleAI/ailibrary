"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

try.py
==============================================================================
"""
import json
import pymongo
from pymongo import MongoClient
import pprint


host = '127.0.0.1'
port = 27017

query = '{"species": "Iris-setosa"}'

d_query = json.loads(query)

client = MongoClient(host=host, port=port)
db = client.get_database('iris')
iris = db.get_collection('iris')

result = iris.find(d_query)

to_json = {}
for record in result:
	k = str(record['_id'])
	del record['_id']
	to_json[k] = record

with open('file.json', 'w') as jf:
	json.dump(to_json, jf)

