import pandas as pd
import os
import sys
import redshift_connector
from cnvrgp import Cnvrg

# import logging
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

def connect(host, db, trusted=False, **kwargs):
    try:
        conn = redshift_connector.connect(
            host=host,
            database=db,
            user=os.environ.get("REDSHIFT_UID"),
            password=os.environ.get("REDSHIFT_PWD"),
            **kwargs
#         ssl_insecure=trusted
        )
        return conn
    except Exception as e:
        print("Could not connect to Redshift cluster, check your parameters")
        print(e)
        sys.exit(1)

def close_connection(conn=None):
    try:
        conn.close()
    except Exception as e:
        print("Could not close connection to Redshift cluster")
        print(e)
        sys.exit(1)

def run(conn=None, query=None,commit=False, **kwargs):
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        if commit:
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.commit()
        else:
            cursor: redshift_connector.Cursor = conn.cursor()
            return cursor.execute(query)

    except Exception as e:
        ###consider implementing command disable, i.e. let the user run read-only commands. ##
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)
        
def to_df(conn=None, query=None, **kwargs):
    cursor = run(conn, query, **kwargs)
    df: pd.DataFrame = cursor.fetch_dataframe()
    return df

def to_csv(conn=None, query=None, filename=None, **kwargs):
    df = to_df(conn, query, **kwargs)
    df.to_csv(filename, index=False)

def upload_to_ds(cnvrg_ds, filename, output_dir):
    if cnvrg_ds.lower() != None:
        os.chdir(output_dir)
        cnvrg = Cnvrg()
        ds = cnvrg.datasets.get(cnvrg_ds)
        try:
            ds.reload()
        except:
            print('The provided Dataset was not found')
            print('Creating a new dataset named {}'.format({cnvrg_ds}))
            ds = cnvrg.datasets.create(name=cnvrg_ds)
        print('Uploading files to Cnvrg dataset')
        ds.put_files([filename])