import time
import argparse
import os
import pandas as pd


from ds_snowflake.con_snowflake import SnowflakeConnector

if __name__ == '__main__':
    snf = SnowflakeConnector(user="LEAH", password="Cnvrg18!", warehouse="COMPUTE_WH",
                                           account="au77965.west-us-2.azure", database="SNOWFLAKE_SAMPLE_DATA",
                                           schema="TPCDS_SF100TCL")

    results = snf.run(query="Select CC_CALL_CENTER_ID from CALL_CENTER").fetchall()
    for rec in results:
        print(rec)
    snf.close_connection()