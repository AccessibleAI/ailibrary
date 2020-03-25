"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

postgresql.py
==============================================================================
"""
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Connect to a PostgreSQL database')
	parser.add_argument('--sql', help='SQL query', default='')
	parser.add_argument('--database', help='', default='users')
	parser.add_argument('--host', help='', default='127.0.0.1')
	parser.add_argument('--port', help='', default='5432')
	parser.add_argument('--pg_env', help='env variable for password', default='PG_PW')
	parser.add_argument('--password_env', help='env variable for password', default='PG_PW')
	parser.add_argument('--username_env', help='env variable for username', default='PG_UN')
	args = parser.parse_args()



