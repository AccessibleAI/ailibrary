"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import os
import pymongo


class MongoDBConnector:

	def __init__(self, credentials):
		self.__successfully_connected = False
		self.__host = credentials.host
		self.__port = int(credentials.port)
		self.__query = credentials.query
		self.__db_name = credentials.db_name
		self.__table_name = credentials.table_name
		self.__cnvrg_ds = credentials.cnvrg_ds
		self.__output_csv = '{db_name}_{table_name}.csv'.format(db_name=self.__db_name,
																table_name=self.__table_name)
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
			conn = psycopg2.connect("host={host} port={port} dbname={db_name}".format(
																					host=self.__host,
																					port=self.__port,
																					db_name=self.__db_name))
			self.__connection = conn
			self.__successfully_connected = True
			print("Connection has been done successfully!")
		except:
			print("Connection to database failed")
			print("terminating...")

	def __terminate_conn(self):
		self.__connection.close()

	def __run_query(self):
		# Create a cursor object
		cur = self.__connection.cursor()
		print('Running query: ', self.__query)

		copy_to_csv_query = "COPY (" + self.__query + ") TO STDOUT WITH CSV DELIMITER ','"
		with open("{}".format(self.__output_csv), "w") as file:
			print('Saving query results to {}'.format(self.__output_csv))
			print('cnvrg_tag_data_path: {}'.format(self.__output_csv))
			cur.copy_expert(copy_to_csv_query, file)
		cur.close()

	def __push_to_app(self):
		os.system('cnvrg data put {url} {exported_file}'.format(url=self.__cnvrg_ds, exported_file=self.__output_csv))

	def visualize(self):
		pass
