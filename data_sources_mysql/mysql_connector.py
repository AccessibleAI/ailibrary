"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

connector.py.py
==============================================================================
"""
import os

import cnvrg
import pandas as pd
import mysql.connector

from mysql.connector import errorcode, InterfaceError
from cnvrg import Experiment
from cnvrg.charts import Heatmap

# The default size of head when visualizing.
DEF_HEAD = 200

class MySQLConnector:
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
		username, password = MySQLConnector.__extract_username_and_password()
		if username is None and password is None:
			self.__successfully_connected = False
			return

		try:
			print('Trying to connect to database')
			conn = mysql.connector.connect(
										host=self.__host,
										port=self.__port,
										database=self.__db_name,
										user=username,
										passwd=password)

			self.__connection = conn
			self.__successfully_connected = True
			print("Connection has been done successfully!")

			return self.__successfully_connected
		except mysql.connector.Err as err:
			if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
				print("Something is wrong with your username or password.")
			elif err.errno == errorcode.ER_BAD_DB_ERROR:
				print("Database does not exist.")
			else:
				print(err)

			print("Connection to database failed. \nterminating... ")
			self.__successfully_connected = False
			return

	def __terminate_conn(self):
		self.__connection.close()

	def __run_query(self):
		# Create a cursor object
		cur = self.__connection.cursor()
		print('Running query: ', self.__query)

		cur.execute(self.__query)

		try:
			all_columns = cur.column_names
			all_rows = cur.fetchall()
		except InterfaceError:
			print("No results for the given query.")
			return

		print('Saving query results to {}'.format(self.__output_csv))
		print('cnvrg_tag_data_path: {}'.format(self.__output_csv))

		with open("{}".format(self.__output_csv), "w") as file:
			file.write(all_columns) 			# Columns.
			file.writelines(all_rows)           # Rows.

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
		head_as_heatmap = MySQLConnector.__helper_plot_confusion_matrix(head)
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