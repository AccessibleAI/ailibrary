from cnvrg import Endpoint
from cnvrg import Dataset
import time
import csv
import argparse
import os

try:
    parser = argparse.ArgumentParser(description='set input arguments')
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
        print("input file is empty")
        exit(0)

    #fetch endpoint details
    endpoint = Endpoint(slug)
    if endpoint is None:
        print("Can't find Endpoint {endpoint_id}").format(endpoint_id=slug)
        exit(1)
        
    #fetch dataset details
    ds = Dataset(dataset)
    if ds is None:
        print("Can't find Dataset {dataset}").format(dataset=dataset)
        exit(1)
    ds_url = ds.get_full_url()

    endpoint.link_experiment()

    print("Endpoint is scaling up")
    endpoint.scale_up()
    
    is_running = endpoint.is_deployment_running()
    while not is_running:
        print("Endpoint is not running yet waiting 10 seconds")
        time.sleep(10)
        is_running = endpoint.is_deployment_running()
    print("Endpoint is online starting batch predict")
    
    time.sleep(20)
    
    ## Input file should be absulut path
    row_list=[]
    with open(input_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            try:
                resp = endpoint.predict(row)
                row_list.append([row, resp.get("prediction")])
            except Exception as e:
                print(e)
    
    ## create output file tree if not exists
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    ## Output file should be absulut path in /cnvrg
    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        for row in row_list:
            writer.writerow(row)

    print("uploading {output_csv} file to dataset {dataset_slug}".format(output_csv=output_file, dataset_slug=dataset))

    os.system('cnvrg data put {url} {exported_file}'.format(url=ds_url, exported_file=output_file))

    print("Batch predict has finished, scaling down endpoint")
    endpoint.scale_down()

except Exception as e:
    if endpoint:
        endpoint.scale_down()
    print(e)
    exit(1)
