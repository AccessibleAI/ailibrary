"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

database_connection.py
==============================================================================
"""
import os
import time
import psycopg2  # installed psycopg2-binary

WAITING = 10

def connect_and_save(args):
	host = args.host
	port = int(args.port)
	query = args.query
	db_name = args.db_name
	table_name = args.table_name
	cnvrg_ds = args.cnvrg_ds

	output_csv = '{db_name}_{table_name}.csv'.format(db_name=db_name, table_name=table_name)

	# password = os.environ[args.password_env]  #'123456789'
	# username =  os.environ[args.username_env]   #'omerliberman'

	try:
		print('Trying to connect to database')
		conn = psycopg2.connect("host={host} port={port} dbname={db_name}".format(host=host, port=port, db_name=db_name))
	except:
		time.sleep(WAITING)
		print("Connection to database failed")
		time.sleep(WAITING)
		print("terminating...")
		return

	time.sleep(WAITING)
	print("Connection has been done successfully!")

	# Create a cursor object
	cur = conn.cursor()
	print('Running query: ', query)

	copy_to_csv_query = "COPY (" + query + ") TO STDOUT WITH CSV DELIMITER ','"
	with open("{}".format(output_csv), "w") as file:
		print('Saving query results to {}'.format(output_csv))
		print('cnvrg_tag_data_path: {}'.format(output_csv))
		cur.copy_expert(copy_to_csv_query, file)

	# uploading data to cnvrg:
	os.system('cnvrg data put {url} {exported_file}'.format(url=cnvrg_ds, exported_file=output_csv))

	time.sleep(WAITING)
	print('Terminating connection')
	cur.close()
	conn.close()


def visualize():
	pass
