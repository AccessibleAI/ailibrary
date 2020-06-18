import argparse
import pandas as pd
import snowflake.connector


class SnowflakeConnector():

    def __init__(self, user=None, password=None, account=None, warehouse=None, database=None, schema=None):
        self.user = user
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.cur =None
        self.connect()


    def connect(self):
        try:
            conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            self.cur=conn.cursor()
        except Exception as e:
            print("Could not connect to snowflake, check your parameters")
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

    def to_df(self):
        df = pd.DataFrame.from_records(iter(self.cur), columns=[x[0] for x in self.cur.description])
        return df

    def to_csv(self, query, file_name):
        cur = self.run(query)
        col_headers = [i[0] for i in cur.description]
        rows = [list(i) for i in cur.fetchall()]
        df = pd.DataFrame(rows, columns=col_headers)

        df.to_csv(file_name, index=False)