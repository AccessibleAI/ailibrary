"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import os
import psycopg2

def connect_and_save(args):
	host = args.host
	port = int(args.port)
	sql = args.sql
	db = args.database

	password = os.environ[args.password_env]
	username = os.environ[args.username_env]

	print('Connecting to DB')
	conn = psycopg2.connect(host=host, port = port, database=db, user=username, password=password)

	# Create a cursor object
	cur = conn.cursor()
	print('Running query: ', sql)

	copy_sql = "COPY (" + sql + ") TO STDOUT WITH CSV DELIMITER ','"
	with open("data.csv", "w") as file:
		print('Saving to data.csv: ')
		print('cnvrg_tag_data_path: data.csv')
		cur.copy_expert(copy_sql, file)


	print('Closing connection')
	cur.close()
	conn.close()
