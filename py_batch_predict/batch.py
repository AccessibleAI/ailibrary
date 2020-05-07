from cnvrg import Endpoint
import time
import csv
import argparse

try:
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--endpoint_id', action="store", dest='slug', type=str, default=60)
    parser.add_argument('--input_file', action="store", dest='input', type=str, default=60)
    parser.add_argument('--output_file', action="store", dest='output', type=str, default=60)
    
    args = parser.parse_args()
    slug = args.slug
    input_file = args.input
    output_file = args.output
    
    e = Endpoint(slug)
    
    e.scale_up()
    
    is_running = e.is_deployment_running()
    while not is_running:
        print("Endpoint is not running yet waiting 10 seconds")
        time.sleep(10)
        is_running = e.is_deployment_running()
    print("Endpoint is online starting batch predict")
    
    row_list=[]
    
    time.sleep(20)
    
    ## Input file should be absulut path
    with open(input_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            try:
                resp = e.predict(row)
                row_list.append([row, resp.get("prediction")])
            except Exception as e:
                print(e)
    
    ## Output file should be absulut path in /cnvrg
    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(row_list)
    
    print("Batch predict has finished scaling down endpoint")
    e.scale_down()
except Exception as e:
    e.scale_down()
    print(e)
    exit(1)
