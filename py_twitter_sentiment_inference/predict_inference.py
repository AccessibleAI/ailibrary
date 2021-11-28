# from cnvrg import Endpoint
from cnvrgv2 import Cnvrg, EndpointKind, Project
from cnvrg import Dataset
import time
import csv
import argparse
import os
import pandas as pd
import json
import requests

try:
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--input_file', action="store", dest='input', type=str, default='')
    parser.add_argument('--output_file', action="store", dest='output', type=str, default='')
    parser.add_argument('--dataset', action="store", dest='dataset', type=str, default='')

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    dataset = args.dataset
 
    ## checking that input file exists and not empty otherwise theres no point to scale up the endpoint
    f = open(input_file, "r")
    if os.path.getsize(input_file) == 0:
        print('Input file: {input_file} is empty,  Aborting'.format(input_file=input_file))
        exit(1)

    #fetch endpoint details
    c = Cnvrg()
    proj = Project()#Endpoint(slug)
    endpoints_list= list(proj.endpoints.list())
    inference_exists = False
    for e in endpoints_list:
        if e.title == 'twitter-inference':
            inference_exists = True
            endpoint = e
            break
    if not inference_exists:
        print('Creating new inference endpoint')
        endpoint = proj.endpoints.create(title=f'twitter-inference',
                                         file_name='predict_sentiment.py',
                                         function_name='predict',
                                         kind=EndpointKind.BATCH)

    #fetch dataset details
    ds = Dataset(dataset)
    if ds is None:
        print('Can\'t find Dataset {dataset}'.format(dataset=dataset))
        exit(1)
    ds_url = ds.get_full_url()

#     endpoint.link_experiment()

    print("Starting to scale up endpoint")
    time.sleep(10)
    
    is_running = endpoint.batch_is_running()
    if not inference_exists:
        endpoint.reload()
    if not is_running:
        endpoint.batch_scale_up()
    while not is_running:
        print("Endpoint is not running yet, retrying in 10 seconds")
        time.sleep(10)
        is_running = endpoint.batch_is_running()
    print("Endpoint is online, starting batch prediction")
    
    time.sleep(20)

    ## Input file should be absulut path
    row_list = []
    data = pd.read_csv(input_file, header=0)
    headers = {
            'Cnvrg-Api-Key': endpoint.api_key,
            'Content-Type': 'application/json',
        }
    for row in data.values:
        try:
            r_list = row.tolist()
            resp = endpoint.predict(r_list)
            row_list.append([r_list, resp])
        except Exception as e:
            print(e)

    ## create output file tree if not exists
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    ## Output file should be absulut path in /cnvrg
    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["input", "prediction"])
        for row in row_list:
            writer.writerow(row)

    print('Uploading {output_file} to dataset {dataset}'.format(output_file=output_file, dataset=dataset))
    os.system('cnvrg data put {url} {exported_file}'.format(url=ds_url, exported_file=output_file))

    print("Batch prediction has finished, scaling down")
    endpoint.batch_scale_down()

except Exception as e:
    if endpoint:
        endpoint.batch_scale_down()
    print(e)
    exit(1)