
#  BigQuery Connector
BigQuery Connector library provides an easy way to query different databases using Google Cloud BigQuery API.

## How It Works

Using BigQuery API to receive query results

## Running

1) Define a new secret in your cnvrg project:
Go to **Settings** -> **Secrets** -> **Add**. Set the secret's name to be ```GOOGLE_APPLICATION_CREDENTIALS```, and the value to be the path of the JSON file that contains your service account key


2) Choose the Bigquery AI Library component, and pass the query and arguments you'd wish to use.

## Running in interactive mode (Notebooks / IDE)
####Loading the library

<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;"><pre><code class='python'>from cnvrg import Library
library = Library('cnvrg/bigquery_connector')
library.load()
conn = library.run()</code></pre></div>

#### running the library
<div style="background:#f7fbff; font-size:14px; padding:10px 10px 10px 10px;">
<pre><code class='python'>
conn = library.run()
result = conn.query("SELECT * FROM table")</code></pre></div>

## Demo Inputs

#### Get BigQuery connector:
```
--query "SELECT * FROM ..."
```

##### Get Dataframe of the big query result:
```
--query "SELECT * FROM ... --df=true"
```

#### Export query results to file:
```
--query "SELECT * FROM ... --csv=true --filename= file.csv --output_dir=/cnvrg"
```

#### Export query results to dataset:
```
--query "SELECT * FROM ... --cnvrg_ds=dataset_name --filename= file.csv --output_dir=/cnvrg"
```
