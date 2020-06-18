import argparse
import os

from ds_snowflake import con_snowflake




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--user', action="store", dest='user', type=str, default='')
    parser.add_argument('--account', action="store", dest='account', type=str, default='')
    parser.add_argument('--warehouse', action="store", dest='warehouse', type=str, default='')
    parser.add_argument('--database', action="store", dest='database', type=str, default='')
    parser.add_argument('--schema', action="store", dest='schema', type=str, default='')
    parser.add_argument('--password', action="store", dest='password', type=str, default='')
    parser.add_argument('--query', action="store", dest='query', type=str, default='')
    parser.add_argument('--filename', action="store", dest='filename', type=str, default='')
    args = parser.parse_args()

    # get confidentail parameters from args or os
    password = os.getenv('SNOWFLAKE_PASSWORD') or args.password
    warehouse = os.getenv('SNOWFLAKE_WAREHOUSE') or args.warehouse
    account = os.getenv('SNOWFLAKE_Account') or args.account
    query = args.query
    snf = con_snowflake.SnowflakeConnector(user=args.user, password=password, warehouse=warehouse, account=account,database=args.database,schema=args.schema)
    #test
    snf.to_csv(query=query,file_name=args.filename)
    # setup the connection

