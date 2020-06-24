import time
import argparse
import os
import pandas as pd
import importlib



def connect(mod, **kwatgs):
    mod.connect(**kwatgs)
if __name__ == '__main__':

    model_name = "sql_connector"
    module_path =  model_name + ".py"
    spec = importlib.util.spec_from_file_location(model_name,
                                                  module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.environ["SQL_UID"] = "leah"
    os.environ["SQL_PWD"] = "Cnvrg1234!!!!"
    # print(os.environ.get("SQL_UID"))
    connect(mod,driver="ODBC Driver 17 for SQL Server", server="cnvrgtest.database.windows.net", database="test123" )
    #mod.connect(driver="ODBC Driver 17 for SQL Server", server="cnvrgtest.database.windows.net", database="test123")

    #sql = SQLConnector(uid="leah", pwd="Cnvrg1234!!!!", driver="ODBC Driver 17 for SQL Server",
     #                  server="cnvrgtest.database.windows.net", database="test123")
    #query = sql.run("SELECT * FROM users")
    #df = sql.to_df(query="SELECT * FROM users", parse_dates=["name", "email"])
    #sql.to_csv(query="SELECT * FROM users", filename="test.csv")