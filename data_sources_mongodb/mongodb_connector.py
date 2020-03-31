"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import json
import os
import shutil

from pymongo import *


class MongoDBConnector:

	def __init__(self, credentials):
		self.__successfully_connected = False
		self.__host = credentials.host
		self.__port = int(credentials.port)
		self.__db_name = credentials.db_name
		self.__collection_name = credentials.collection
		self.__cnvrg_ds = credentials.cnvrg_ds
		self.__output_json = '{db_name}_{table_name}.json'.format(db_name=self.__db_name,
																table_name=self.__collection_name)
		self.__query = MongoDBConnector.preprocess_query(credentials.query)
		#self.__password = os.environ[credentials.password_env]
		#self.__username =  os.environ[credentials.username_env]

	def run(self):
		self.__connect()
		if self.__successfully_connected:
			self.__run_query()
			self.__push_to_app()
			self.__terminate_conn()

	def __connect(self):
		try:
			print('Trying to connect to database')
			self.__mongo_client = MongoClient(host=self.__host, port=self.__port)
			self.__successfully_connected = True
			print("Connection has been done successfully!")
		except:
			print("Connection to database failed")
			print("terminating...")

	def __terminate_conn(self):
		self.__mongo_client.close()

	def __run_query(self):
		print('Running query: ', self.__query)
		self.__db = self.__mongo_client.get_database(self.__db_name)
		self.__collection = self.__db.get_collection(self.__collection_name)
		result = self.__collection.find(self.__query)

		try:
			os.mkdir('query_results')
		except FileExistsError:
			shutil.rmtree('query_results')
			os.mkdir('query_results')

		self.__jsons_created = False
		results_counter = 0
		for record in result:
			record['_id'] = str(record['_id'])
			with open('query_results/{}.json'.format(record['_id']), 'w') as jf:
				json.dump(record, jf)
			results_counter += 1
			self.__jsons_created = True
		print('{} results for the query.'.format(results_counter))

	def __push_to_app(self):
		if self.__jsons_created:
			print('Saving query results to {}'.format(self.__cnvrg_ds))

			os.chdir('query_results')
			os.system('cnvrg data put {url} *.json '.format(url=self.__cnvrg_ds))
			os.chdir('..')
			shutil.rmtree('query_results')
		else:
			print("No result for query.")

	def visualize(self):
		pass

	@staticmethod
	def preprocess_query(query):
		colon = query[1: -1].find(':')
		left_paren = query.find('{')
		right_paren = query.find('}')
		key = query[left_paren + 1 : colon + 1]
		value = query[colon + 2: right_paren]
		return { key : value }