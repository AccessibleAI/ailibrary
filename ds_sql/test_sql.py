import time
import argparse
import os
import pandas as pd


from ds_sql.con_sql import SQLConnector

if __name__ == '__main__':

    sql = SQLConnector(uid="leah", pwd="Cnvrg1234!!!!", driver="ODBC Driver 17 for SQL Server",
                       server="cnvrgtest.database.windows.net", database="test123")
    query = sql.run("SELECT * FROM users")
    df = sql.to_df(query="SELECT * FROM users", parse_dates=["name", "email"])
    sql.to_csv(query="SELECT * FROM users", filename="test.csv")