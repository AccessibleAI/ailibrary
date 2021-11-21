import os
import argparse

parser = argparse.ArgumentParser(description="""Kaggle Dataset Connector""")
parser.add_argument('--dataset_name', action='store', dest='dataset_name', required=True, help="""--- The name of the dataset ---""")

parser.add_argument('--dataset_path', action='store', dest='dataset_path', default='/cnvrg/output', help="""--- The path to save the dataset files to ---""")

parser.add_argument('--cnvrg_dataset', action='store', dest='cnvrg_dataset', required=False, default='None', help="""--- the name of the cnvrg dataset to store in ---""")

parser.add_argument('--file_name', action='store', dest='file_name', required=False, default='None', help="""--- If a single file is needed then this is the name of the file ---""")

parser.add_argument('--project_dir', action='store', dest='project_dir', help="""--- For inner use of cnvrg.io ---""")

parser.add_argument('--output_dir', action='store', dest='output_dir', help="""--- For inner use of cnvrg.io ---""")

args = parser.parse_args()
dataset_name = args.dataset_name
dataset_path = args.dataset_path
file_name = args.file_name
cnvrg_dataset = args.cnvrg_dataset

download_command = f'kaggle datasets download {dataset_name} --unzip'

if dataset_path:
    download_command +=  f' -p {dataset_path}'
if file_name.lower() != 'none':
    download_command += f' -f {file_name}'

print(f'Downloading dataset {dataset_name} to {dataset_path}')
os.system(download_command)

if cnvrg_dataset.lower() != 'none':
    from cnvrgp import Cnvrg
    cnvrg = Cnvrg()
    ds = cnvrg.datasets.get(cnvrg_dataset)
    try:
        ds.reload()
    except:
        print('The provided Dataset was not found')
    print(f'Creating a new dataset named {cnvrg_dataset}')
    ds = cnvrg.datasets.create(name=cnvrg_dataset)
    print('Uploading files to Cnvrg dataset')
    ds.put_files(paths=[dataset_path])
    
