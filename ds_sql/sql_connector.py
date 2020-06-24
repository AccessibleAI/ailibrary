import pandas as pd
import pyodbc
import csv
import os


def connect(driver=None, server=None, database=None, trusted_connection=False,port=None):
    try:
        uid = os.environ.get("SQL_UID")
        pwd = os.environ.get("SQL_PWD")
        port = (port or 1433)
        config = dict(server=server,
                      port=port,  # change this to your SQL Server port number [1433 is the default]
                      database=database,
                      username=uid,
                      password=pwd)
        if trusted_connection:
            conn_str = ('SERVER={server};' +
                                'DATABASE={database};' +
                                'TRUSTED_CONNECTION=yes')
        else:
            conn_str = ('SERVER={server},{port};' +
                    'DATABASE={database};' +
                    'UID={username};' +
                    'PWD={password}')


        conn = pyodbc.connect(
            r"DRIVER={%s};" % driver +
            conn_str.format(**config)
        )
        conn=conn
        cur=conn.cursor()
        return cur
    except Exception as e:
        print("Could not connect to SQL server, check your parameters")
        print(e)
        exit(1)
def close_connection(self):
    try:
        self.cur.close()
    except Exception as e:
        print("Could not close connection to snowflake")
        print(e)
        exit(1)
def run(cur, query=None):
    if query is None:
        print("Query can't be empty")
        exit(1)
    try:
        return cur.execute(query)
    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        exit(1)

def to_df(conn, query,index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None):
    df = pd.read_sql_query(query, conn,index_col, coerce_float, params, parse_dates, chunksize )
    return df

def to_csv(query, file_name):
    cur = run(query)
    col_headers = [i[0] for i in cur.description]
    rows = [list(i) for i in cur.fetchall()]
    df = pd.DataFrame(rows, columns=col_headers)

    df.to_csv(file_name, index=False)

