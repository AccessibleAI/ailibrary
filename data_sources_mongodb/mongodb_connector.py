"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import json
import os
from pymongo import *


class MongoDBConnector:

	def __init__(self, credentials):
		self.__successfully_connected = False
		self.__host = credentials.host
		self.__port = int(credentials.port)
		self.__db_name = credentials.db_name
		self.__collection = credentials.__collection
		self.__cnvrg_ds = credentials.cnvrg_ds
		self.__output_json = '{db_name}_{table_name}.json'.format(db_name=self.__db_name,
																table_name=self.__collection)
		#self.__password = os.environ[credentials.password_env]
		#self.__username =  os.environ[credentials.username_env]
		try:
			self.__query = json.loads(credentials.query)
		except json.decoder.JSONDecodeError:
			raise ValueError("Bad format of query. Should be: """"{ "key" : "value" }""""")

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
		collection = self.__db.get_collection(self.__collection)

		with open("{}".format(self.__output_json), "w") as file:
			print('Saving query results to {}'.format(self.__output_csv))
			print('cnvrg_tag_data_path: {}'.format(self.__output_csv))
			cur.copy_expert(copy_to_csv_query, file)
		cur.close()

	def __push_to_app(self):
		os.system('cnvrg data put {url} {exported_file}'.format(url=self.__cnvrg_ds, exported_file=self.__output_csv))

	def visualize(self):
		pass
