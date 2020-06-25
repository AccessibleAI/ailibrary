import pandas as pd
import snowflake.connector
import os
import sys

def connect(account=None, warehouse=None, database=None, schema=None):
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    account = account or os.environ.get("SNOWFLAKE_ACCOUNT")
    warehouse = warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE")
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        return conn
    except Exception as e:
        print("Could not connect to snowflake, check your parameters")
        print(e)
        sys.exit(1)



def close_connection(cur=None):
    try:
        cur.close()
    except Exception as e:
        print("Could not close connection to snowflake")
        print(e)
        sys.exit(1)
def run(cur=None, query=None):
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        return cur.execute(query)
    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)

def to_df(cur=None):
    df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
    return df

def to_csv(cur=None, query=None, file_name=None):
    cur = run(cur=cur, query=query)
    col_headers = [i[0] for i in cur.description]
    rows = [list(i) for i in cur.fetchall()]
    df = pd.DataFrame(rows, columns=col_headers)

    df.to_csv(file_name, index=False)