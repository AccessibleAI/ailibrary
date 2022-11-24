from cnvrg import Endpoint
from cnvrg import Dataset
import time
import csv
import argparse
import os
import pandas as pd

try:
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--project_dir', action='store', dest='project_dir',help="""--- For inner use of cnvrg.io ---""")
    parser.add_argument('--output_dir', action='store', dest='output_dir',help="""--- For inner use of cnvrg.io ---""")
    parser.add_argument('--endpoint_id', action="store", dest='slug', type=str, default='')
    parser.add_argument('--input_file', action="store", dest='input', type=str, default='')
    parser.add_argument('--output_file', action="store", dest='output', type=str, default='')
    parser.add_argument('--dataset', action="store", dest='dataset', type=str, default='')

    args = parser.parse_args()
    slug = args.slug
    input_file = args.input
    output_file = args.output
    dataset = args.dataset
    
    ## checking that input file exists and not empty otherwise theres no point to scale up the endpoint
    f = open(input_file, "r")
    if os.path.getsize(input_file) == 0:
        print('Input file: {input_file} is empty,  Aborting'.format(input_file=input_file))
        exit(1)

    #fetch endpoint details
    endpoint = Endpoint(slug)
    endpoint_data = endpoint.data
    if endpoint_data is None:
        print('Can\'t find Endpoint {slug}'.format(slug=slug))
        exit(1)
        
    #fetch dataset details
    ds = Dataset(dataset)
    if ds is None:
        print('Can\'t find Dataset {dataset}'.format(dataset=dataset))
        exit(1)
    ds_url = ds.get_full_url()

    endpoint.link_experiment()

    print("Starting to scale up endpoint")
    endpoint.scale_up()
    
    is_running = endpoint.is_deployment_running()
    while not is_running:
        print("Endpoint is not running yet, retrying in 10 seconds")
        time.sleep(10)
        is_running = endpoint.is_deployment_running()
    print("Endpoint is online, starting batch prediction")
    
    time.sleep(20)

    ## Input file should be absulut path
    row_list = []
    data = pd.read_csv(input_file, header=0)
    for row in data.values:
        try:
            r_list = row.tolist()
            resp = endpoint.predict(r_list)
            row_list.append([r_list, resp.get("prediction")])
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
    endpoint.scale_down()

except Exception as e:
    if endpoint:
        endpoint.scale_down()
    print(e)
    exit(1)
