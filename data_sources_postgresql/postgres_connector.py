"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import os

import cnvrg
import pandas as pd
import psycopg2  # installed psycopg2-binary
from cnvrg.charts import Heatmap
from cnvrg import Experiment

DEF_HEAD = 200


class PostgresConnector:
	def __init__(self, credentials):
		self.__successfully_connected = False
		self.__host = credentials.host
		self.__port = int(credentials.port)
		self.__query = credentials.query
		self.__db_name = credentials.db_name
		self.__table_name = credentials.table_name
		self.__ds_name = credentials.cnvrg_ds  # just the name of the ds.
		self.__output_csv = '{db_name}_{table_name}.csv'.format(
			db_name=self.__db_name,
			table_name=self.__table_name)
		self.__vis = (credentials.visualize == 'True')

		self.__cnvrg_env = True
		try:
			self.__experiment = Experiment()
		except cnvrg.modules.UserError:
			self.__cnvrg_env = False

	def run(self):
		self.__connect()
		if self.__successfully_connected:
			self.__run_query()
			self.__push_to_app()
			self.__terminate_conn()

			if self.__vis:
				self.__visualize()

	def __connect(self):
		username, password = PostgresConnector.__extract_username_and_password()
		if username is None and password is None:
			self.__successfully_connected = False
			return

		try:
			print('Trying to connect to database')
			conn = psycopg2.connect(
				"host={host} port={port} dbname={db_name} "
				"user={username} password={password}".format(
					host=self.__host,
					port=self.__port,
					db_name=self.__db_name,
					username=username,
					password=password))
			self.__connection = conn
			self.__successfully_connected = True
			print("Connection has been done successfully!")
			return self.__successfully_connected
		except:
			print("Connection to database failed")
			print("terminating...")
			self.__successfully_connected = False
			return

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
		try:
			cnvrg_owner = os.environ['CNVRG_OWNER']
		except KeyError:
			return

		url = "https://app.cnvrg.io/{cnvrg_owner}/datasets/{ds_name}".format(cnvrg_owner=cnvrg_owner, ds_name=self.__ds_name)
		os.system('cnvrg data put {url} {exported_file}'.format(url=url, exported_file=self.__output_csv))

	def __visualize(self):
		df = pd.read_csv(self.__output_csv)
		rows = min(DEF_HEAD, len(df))
		head = df.head(rows)
		head_as_heatmap = PostgresConnector.__helper_plot_confusion_matrix(head)
		if self.__cnvrg_env:
			self.__experiment.log_chart("sample ({num_of_rows} rows)".format(num_of_rows=rows), data=Heatmap(z=head_as_heatmap))
		else:
			print(head_as_heatmap)

	@staticmethod
	def __helper_plot_confusion_matrix(confusion_matrix, mat_x_ticks=None, mat_y_ticks=None, digits_to_round=3):
		"""
		:param confusion_matrix: the values in the matrix.
		:param mat_x_ticks, mat_y_ticks: ticks for the axis of the matrix.
		"""
		output = []
		for y in range(len(confusion_matrix)):
			for x in range(len(confusion_matrix[y])):
				x_val = x if mat_x_ticks is None else mat_x_ticks[x]
				y_val = y if mat_y_ticks is None else mat_y_ticks[y]
				output.append((x_val, y_val, round(float(confusion_matrix[x][y]), digits_to_round)))
		return output

	@staticmethod
	def __extract_username_and_password():
		"""
		Returns the values of the environment variables.
		If both exist -> returns (user_name, password).
		Otherwise -> returns (None, None).
		"""
		password = os.environ.get("PG_PASSWORD")
		username = os.environ.get("PG_USERNAME")

		if username is None or password is None:
			print(
				"One of the required environment "
				"variables is not defined correctly.")
			return None, None

		return username, password
