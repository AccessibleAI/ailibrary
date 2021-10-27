#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import auth
import asyncio
from cnvrgp import Cnvrg


##############################################################################
# Parses the command line arguments
##############################################################################
def parse_parameters():
    """Command line parser."""
    # epilog message: Custom text after the help
    epilog = """
    Example of use:
        python3 %(prog)s --batch_upload --container_name="cnvrg-container" --output="/path/to/dir"
        python3 %(prog)s --upload --container_name="cnvrg-container" --output="/path/to/dir" --file_name="file.jpg"
        python3 %(prog)s --batch_download --prefix=".png" --container_name="cnvrg-container" --output="/path/to/dir"
        python3 %(prog)s --download --container_name="cnvrg-container" --output="/path/to/dir" --file_name="file.jpg"
        python3 %(prog)s --batch_download --prefix=".png" --container_name="cnvrg-container" --output="/path/to/dir" --cnvrg_dataset="name"
    """
    # Create the argparse object and define global options
    parser = argparse.ArgumentParser(
        description="AzureBlobStorage sample script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    # Add subcommands options
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    actions = subparsers.add_parser(name="actions parser")
    actions.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        help="Download recursively all files with a prefix.",
    )
    actions.add_argument(
        "-over", "--overwrite", action="store_true", dest="overwrite", help="overwrite files when uploading")
    actions.add_argument(
        "-co",
        "--container_name",
        dest="container_name",
        help="container_name",
    )
    actions.add_argument(
        "-o",
        "--output",
        dest="output",
        default=os.environ.get('CNVRG_WORKDIR'),
        help="Download to a specific location, default is Cnvrg Workdir",
    )
    actions.add_argument(
        "-f",
        "--file_name",
        dest="file_name",
        help="Download blob by filename",
    )
    # cnvrg_dataset
    actions.add_argument('--cnvrg_dataset', "-dataset", help="""--- the name of the cnvrg dataset to store in ---""")
    # Only 1 Action is required at least, and in must.
    actions_group = actions.add_mutually_exclusive_group(required=True)
    actions_group.add_argument(
        "-d", "--download", action="store_true", dest="download", help="Download a specific file")
    actions_group.add_argument(
        "-bd", "--batch_download", action="store_true", dest="batch_download", help="Download a specific folder")
    actions_group.add_argument(
        "-u", "--upload", action="store_true", dest="upload", help="upload a specific file")
    actions_group.add_argument(
        "-bu", "--batch_upload", action="store_true", dest="batch_upload", help="upload a specific folder")





    # If there is no parameter, print help
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    
    return actions.parse_args()


def msg(color, msg_text, exitcode=0, *, end="\n", flush=True, output=None):
    """
    Print colored text.

    Arguments:
        color          (str): color name (blue, red, green, yellow,
                              cyan or nocolor)
        msg_text       (str): text to be printed
        exitcode  (int, opt): Optional parameter. If exitcode is different
                              from zero, it terminates the script, i.e,
                              it calls sys.exit with the exitcode informed

    Keyword arguments (optional):
        end            (str): string appended after the last char in "msg_text"
                              default a newline
        flush   (True/False): whether to forcibly flush the stream.
                              default True
        output      (stream): a file-like object (stream).
                              default sys.stdout

    Example:
        msg("blue", "nice text in blue")
        msg("red", "Error in my script. terminating", 1)
    """
    color_dic = {
        "blue": "\033[0;34m",
        "red": "\033[1;31m",
        "green": "\033[0;32m",
        "yellow": "\033[0;33m",
        "cyan": "\033[0;36m",
        "resetcolor": "\033[0m",
    }

    if not output:
        output = sys.stdout

    if not color or color == "nocolor":
        print(msg_text, end=end, file=output, flush=flush)
    else:
        if color not in color_dic:
            raise ValueError("Invalid color")
        print(
            "{}{}{}".format(color_dic[color], msg_text, color_dic["resetcolor"]),
            end=end,
            file=output,
            flush=flush,
        )

    if exitcode:
        sys.exit(exitcode)




##############################################################################
# Command to download objects
##############################################################################
def upload_to_dataset(args):

    if args.cnvrg_dataset.lower() != 'none':
        cnvrg = Cnvrg()
        ds = cnvrg.datasets.get(args.cnvrg_dataset)
    try:
        ds.reload()
        if args.output:
           ds.put_files(paths=[args.output])
    except:
        print('The provided Dataset was not found')
        print(f'Creating a new dataset named {args.cnvrg_dataset}')
        ds = cnvrg.datasets.create(name=args.cnvrg_dataset)
        print('Uploading files to Cnvrg dataset')
        if args.output:
           ds.put_files(paths=[args.output])



async def download_blob(file_name, output ,blob):
   # Get full path to the file
   download_file_path = os.path.join(output, file_name)
   with open(download_file_path, "wb") as file:
       stream = await blob.download_blob()
       data = await stream.content_as_bytes()
       file.write(data)



async def download_all_blobs_in_container(azure,output,container_name,prefix):
    coros_list = []
    my_blobs = azure.get_container_client(container_name).list_blobs()
    container = azure.get_container_client(container_name)
    async for blob in my_blobs:
        blobf = container.get_blob_client(blob.name)
        if prefix:
            if prefix in blob.name:
                coros_list.append(asyncio.create_task(download_blob(blob.name,output,blobf)))
        else:
            coros_list.append(asyncio.create_task(download_blob(blob.name,output,blobf)))
    await asyncio.gather(*coros_list)



async def upload_blob(file_name, output ,container_name,azure):
   # Get full path to the file
   try:
       container = azure.get_container_client(container_name)
       output_file_path = os.path.join(output, file_name)
       with open(output_file_path, "rb") as data:
           await container.upload_blob(name=file_name,data=data,overwrite=True)
   except:
       print("FileAlreadyExist")

async def upload_blob_helper(file_name, data ,container,overwrite):
   # Get full path to the file
   try:
           await container.upload_blob(name=file_name,data=data,overwrite=overwrite)
   except:
       print("{} FileAlreadyExist Skipping, to overwrite use : --overwrite".format(file_name))

async def read_files(dir):
    with os.scandir(dir) as files:
        for filename in files:
            if filename.is_file() and not filename.name.startswith('.'):
                yield filename

async def upload_all_blobs_to_container(azure,output,container_name,prefix,overwrite):
    coros_list = []
    container = azure.get_container_client(container_name)
    async with container:
      async for file in read_files(output):
            with open(file.path, "rb") as data:
                if prefix:
                    if prefix in file.name:
                        coros_list.append(asyncio.create_task(upload_blob_helper(file.name,data=data,container=container,overwrite=overwrite)))
                else:
                    coros_list.append(asyncio.create_task(upload_blob_helper(file.name,data=data,container=container,overwrite=overwrite)))
                await asyncio.gather(*coros_list)

##############################################################################
# Main function
##############################################################################
def main():
    """Command line execution."""
    global log

    # Parser the command line
    args = parse_parameters()
    loop = asyncio.get_event_loop()
    try:
        config = auth.Auth()
        if os.getenv("AZURE_STORAGE_ACCESS_KEY"):
            azure = config.auth_shared_key()
        elif os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
            azure = config.auth_connection_string()
        else:
            msg("red","You must define one of the envs auth method",1)

    except ValueError as error:
        msg("red", str(error), 1)

    if args.batch_download:
        print("Starting batch download")
        if args.output:
            if not os.path.isdir(args.output):
                msg("red", "Error: Directory '{}' not found".format(args.output), 1)
        loop.run_until_complete(download_all_blobs_in_container(azure, args.output, args.container_name, args.prefix))
        print("batch download completed")

        #upload to cnvrg dataset
        if args.cnvrg_dataset.lower() != 'none':
            upload_to_dataset(args)

    if args.download:
        print("Starting download")
        container = azure.get_container_client(args.container_name)
        blob = container.get_blob_client(args.file_name)
        loop.run_until_complete(download_blob(args.file_name,args.output,blob))
        print("Download completed")

    if args.batch_upload:
        print("Starting batch upload")
        if args.output:
            if not os.path.isdir(args.output):
                msg("red", "Error: Directory '{}' not found".format(args.output), 1)
        loop.run_until_complete(upload_all_blobs_to_container(azure, args.output, args.container_name, args.prefix,args.overwrite))
        print("Batch upload completed")

    if args.upload:
        print("Starting upload")
        container = azure.get_container_client(args.container_name)
        blob = container.get_blob_client(args.file_name)
        loop.run_until_complete(upload_blob(args.file_name,args.output,args.container_name,azure))
        print("Upload completed")
   

##############################################################################
# Run from command line
##############################################################################
if __name__ == "__main__":
    main()
