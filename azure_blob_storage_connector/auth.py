# coding: utf-8

"""
USAGE:
    auth.py
    Set the environment variables with your own values before running the auth:
    1) AZURE_STORAGE_ACCOUNT_NAME - the name of the storage account
    2) AZURE_STORAGE_ACCESS_KEY - the storage account access key

    optional:
        3) AZURE_STORAGE_CONNECTION_STRING - the connection string to your storage account
"""

import os
from azure.storage.blob.aio import BlobServiceClient
class Auth(object):
    url = "https://{}.blob.core.windows.net".format(
        os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    )
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    shared_access_key = os.getenv("AZURE_STORAGE_ACCESS_KEY")


    def auth_connection_string(self):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        return blob_service_client

    def auth_shared_key(self):
        blob_service_client = BlobServiceClient(account_url=self.url, credential=self.shared_access_key)
        return blob_service_client
