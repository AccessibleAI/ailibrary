import argparse
import os

from ds_sql import con_sql




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--uid', action="store", dest='uid', type=str, default='')
    parser.add_argument('--pwd', action="store", dest='pwd', type=str, default='')
    parser.add_argument('--driver', action="store", dest='driver', type=str, default='')
    parser.add_argument('--server', action="store", dest='server', type=str, default='')
    parser.add_argument('--database', action="store", dest='database', type=str, default='')
    parser.add_argument('--trusted_connection', action="store", dest='trusted_connection', type=str, default='')
    parser.add_argument('--port', action="store", dest='port', type=str, default='')
    parser.add_argument('--query', action="store", dest='query', type=str, default='')
    parser.add_argument('--csv', action="store", dest='csv', type=str, default='')
    parser.add_argument('--filename', action="store", dest='filename', type=str, default='')
    parser.add_argument('--df', action="store", dest='df', type=str, default='')



    args = parser.parse_args()

    # get confidentail parameters from args or os
    pwd = os.getenv('SQL_PWD') or args.pwd
    port = (args.port or 1433)

    sql = con_sql.SQLConnector(uid=args.uid, pwd=pwd, driver=args.driver,
                       server=args.server, database=args.database)

    if args.csv:
        sql.to_csv(query=args.query, file_name=args.filename)
        exit(0)
    if args.df:
        df=sql.to_pd(query=args.query)
        #TODO: need to add df params, and save to file vefore exit
        exit(0)