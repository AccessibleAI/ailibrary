"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

postgresql.py
==============================================================================
"""
import argparse
from database_connection import *

"""
COMMAND FOR DEBUGGING:

--sql="SELECT * FROM types;" --database="types" --host="localhost" --port="5432" --cnvrg_dataset_url="https://app.cnvrg.io/AIintern/datasets/from_postgres"
"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database')

	parser.add_argument('--sql_query', action='store', dest='query', required=True,
						help='SQL query')

	parser.add_argument('--database', action='store', dest='db_name', required=True,
						help='')

	parser.add_argument('--table_name', action='store', dest='table_name', required=True,
						help='')

	parser.add_argument('--host', action='store', dest='host', default='127.0.0.1',
						help='')

	parser.add_argument('--port', action='store', dest='port', default='5432',
						help='')

	parser.add_argument('--cnvrg_dataset_url', action='store', dest='cnvrg_ds', required=True,
						help='')

	parser.add_argument('--pg_env', action='store', dest='pg_env', default='PG_PW',
						help='env variable for password')

	parser.add_argument('--password_env', action='store', dest='password_env', default='PG_PW',
						help='env variable for password')

	parser.add_argument('--username_env', action='store', dest='username_env', default='PG_UN',
						help='env variable for username')

	args = parser.parse_args()
	connect_and_save(args)



