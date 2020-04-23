"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

mysql.py
==============================================================================
"""

import argparse
from mysql_connector import MySQLConnector

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database')

	parser.add_argument('--sql_query', action='store', dest='query', required=True,
						help='(String) SQL query to run over the data set.')

	parser.add_argument('--database', action='store', dest='db_name', required=True,
						help='(String) the name of the PostgreSQL.')

	parser.add_argument('--table_name', action='store', dest='table_name', required=True,
						help='(String) the name of the specific table in the data set.')

	parser.add_argument('--host', action='store', dest='host', default='127.0.0.1',
						help='(String) (Default: 127.0.0.1) IP number.')

	parser.add_argument('--port', action='store', dest='port', default='3306',
						help='(int) (Default: 3306) the port number.')

	parser.add_argument('--dataset', action='store', dest='cnvrg_ds', required=True,
						help='(String) the name of the dataset at cnvrg.')

	parser.add_argument('--visualize', action='store', dest='visualize', default='False',
						help='(bool) (Default: False) if true -> displays a sample of the data.')

	args = parser.parse_args()
	connector = MySQLConnector(args)
	connector.run()