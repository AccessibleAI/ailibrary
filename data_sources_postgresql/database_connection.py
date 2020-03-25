"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import os
import psycopg2  # installed psycopg2-binary

cnvrg_env = True if os.environ.get('CNVRG_WORKDIR') is not None else False

def connect_and_save(args):
	host = args.host
	port = int(args.port)
	query = args.sql
	db_name = args.database

	password = os.environ[args.password_env]
	username = os.environ[args.username_env]

	try:
		print('Trying to connect to database ...')
		conn = psycopg2.connect(host=host, port = port, database=db_name, user=username, password=password)
	except:
		print("Connection to database failed ... terminating.")
		return

	print("Connection has been done successfully!")

	# Create a cursor object
	cur = conn.cursor()
	print('Running query: ', query)

	copy_to_csv_query = "COPY (" + query + ") TO STDOUT WITH CSV DELIMITER ','"
	with open("{db_name}.csv".format(db_name=db_name), "w") as file:
		print('Saving query results to {db_name}.csv'.format(db_name=db_name))
		print('cnvrg_tag_data_path: {db_name}.csv'.format(db_name=db_name))
		cur.copy_expert(copy_to_csv_query, file)

	# uploading data to cnvrg:
	if cnvrg_env:
		pass
		## upload data to cnvrg



	print('Terminating connection')
	cur.close()
	conn.close()


def visualize():
	pass
