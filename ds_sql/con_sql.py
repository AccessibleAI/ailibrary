import pandas as pd
import pyodbc
import csv




class SQLConnector():

    def __init__(self, uid=None, pwd=None, driver=None, server=None, database=None, trusted_connection=False,port=None):
        self.uid = uid
        self.pwd = pwd
        self.driver = driver
        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection
        self.port = (port or 1433)
        self.conn = None
        self.cur = None
        self.connect()


    def connect(self):
        try:
            config = dict(server=self.server,
                          port=self.port,  # change this to your SQL Server port number [1433 is the default]
                          database=self.database,
                          username=self.uid,
                          password=self.pwd)
            if self.trusted_connection:
                conn_str = ('SERVER={server};' +
                                    'DATABASE={database};' +
                                    'TRUSTED_CONNECTION=yes')
            else:
                conn_str = ('SERVER={server},{port};' +
                        'DATABASE={database};' +
                        'UID={username};' +
                        'PWD={password}')


            conn = pyodbc.connect(
                r"DRIVER={%s};" % self.driver +
                conn_str.format(**config)
            )
            self.conn=conn
            self.cur=conn.cursor()

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
    def run(self,query=None):
        if query is None:
            print("Query can't be empty")
            exit(1)
        try:
            return self.cur.execute(query)
        except Exception as e:
            print("Could not run query: %s" % query)
            print(e)
            exit(1)

    def to_df(self, query,index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None):
        df = pd.read_sql_query(query, self.conn,index_col, coerce_float, params, parse_dates, chunksize )
        return df

    def to_csv(self, query, file_name):
        cur = self.run(query)
        col_headers = [i[0] for i in cur.description]
        rows = [list(i) for i in cur.fetchall()]
        df = pd.DataFrame(rows, columns=col_headers)

        df.to_csv(file_name, index=False)

