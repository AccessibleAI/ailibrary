import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import urllib
import csv
import os
import sys
import pymysql


def connect(driver=None, server=None, database=None, trusted_connection=False,port=None,pool_recycle=3600,**kwargs):
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
            conn_str = (
                r'SERVER={server},{port};' 
                r'DATABASE={database};' 
                r'TRUSTED_CONNECTION=yes;'
                r'MARS_Connection=yes')
        else:
            conn_str = (
                r'SERVER={server};' 
                r'DATABASE={database};' 
                r'UID={username};' 
                r'PWD={password};'
                r'MARS_Connection=yes')


        if "mysql" in driver.lower(): # support mysql
            conn_string_engine = "mysql+pymysql://%s:%s@%s:%s/%s" %(uid, pwd,server,port,database)
        else:  # currently it's MSSQL
            conn_string_mssql = r"DRIVER={%s};" % driver + conn_str.format(**config)
            conn_string_engine = "mssql+pyodbc:///?odbc_connect=%s" % urllib.parse.quote_plus(conn_string_mssql)

        engine = create_engine(conn_string_engine,pool_pre_ping=True,pool_recycle=pool_recycle,**kwargs)
        conn = engine.raw_connection()
        return conn,engine
    except Exception as e:
        print("Could not connect to SQL server, check your parameters")
        print(e)
        sys.exit(1)
def close_connection(conn=None):
    try:
        conn.close()
    except Exception as e:
        print("Could not close connection to SQL server")
        print(e)
        sys.exit(1)
def run(conn=None, query=None,commit=False):
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        if commit:
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.commit()
        else:
            return conn.cursor().execute(query)

    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)

def to_df(conn=None, query=None,**kwargs):
    df = pd.read_sql_query(query, conn,**kwargs )
    return df

def to_sql(conn=None, df=None,table_name=None, **kwargs):
    try:
        df.to_sql(table_name, conn, **kwargs)
    except Exception as e:
        print("Could not save data")
        print(e)
        sys.exit(1)

def to_csv(conn=None, query=None, filename=None):
    cur = run(conn=conn, query=query)
    col_headers = [i[0] for i in cur.description]
    rows = [list(i) for i in cur.fetchall()]
    df = pd.DataFrame(rows, columns=col_headers)

    df.to_csv(filename, index=False)

