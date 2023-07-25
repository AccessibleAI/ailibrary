from google.cloud import bigquery
import sys
from cnvrgp import Cnvrg

def connect():
    return bigquery.Client()

def run(query=None):
    client = connect()
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        query_job = client.query(query)
        return query_job.result()

    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)

def to_df(query = None):
    client = connect()
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        df = run(query).to_dataframe(create_bqstorage_client=True)
        return df

    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)

def to_csv(filename = "file.csv", output_dir = "/output", query = None):
    client = connect()
    if query is None:
        print("Query can't be empty")
        sys.exit(1)
    try:
        df = run(query).to_dataframe(create_bqstorage_client=True)
        df.to_csv(output_dir + '/' + filename)

    except Exception as e:
        print("Could not run query: %s" % query)
        print(e)
        sys.exit(1)
        
def upload_to_ds(cnvrg_ds = None, query = None, filename = "file.csv", output_dir ="/output"):
    if cnvrg_ds.lower() != None:
        result = to_csv(query)
        df.to_csv(output_dir + '/' + filename)
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