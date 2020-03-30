"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

try.py
==============================================================================
"""
import pymongo
from pymongo import MongoClient
import pprint

client = MongoClient('mongodb://admin:Password1@localhost:27017/iris')
db = client.iris
iris = db.iris
for record in iris.find():
     pprint.pprint(record)