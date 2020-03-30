"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

postgresql.py
==============================================================================
"""
import argparse
from mongodb_connector import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database')

	parser.add_argument('--query', action='store', dest='query', required=True,
						help='(String) SQL query to run over the data set.')

	parser.add_argument('--database', '--db', action='store', dest='db_name', required=True,
						help='(String) the name of the PostgreSQL.')

	parser.add_argument('--collection', action='store', dest='collection', required=True,
						help='(String) the name of the specific collection in the data set.')

	parser.add_argument('--host', action='store', dest='host', default='127.0.0.1',
						help='(String) IP number.')

	parser.add_argument('--port', action='store', dest='port', default='27017',
						help='(int) the port number.')

	parser.add_argument('--cnvrg_dataset_url', action='store', dest='cnvrg_ds', required=True,
						help='(String) url of the data set at cnvrg platform.')

	parser.add_argument('--pg_env', action='store', dest='pg_env', default='PG_PW',
						help='(String) env variable for password')

	parser.add_argument('--password_env', action='store', dest='password_env', default='PG_PW',
						help='(String) env variable for password')

	parser.add_argument('--username_env', action='store', dest='username_env', default='PG_UN',
						help='(String) env variable for username')

	args = parser.parse_args()
	connector = MongoDBConnector(args)
	connector.run()



