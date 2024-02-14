#!/usr/bin/env python3

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime
    
def fetch_s3_catalog(bucket_name):
    s3_cache = []
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name)
    counter = 0
    for page in page_iterator:
        for item in page["Contents"]:
            if item["Key"][-1] == "/":
                item["Key"] = item["Key"][:-1]

            splitted = item["Key"].split("/")
            if "test" in splitted:
                if len(splitted) < 4:
                    print(item)
                    continue
                    
                item["Sensor"] = splitted[0]
                item["Protocol"] = splitted[1]
                temp_timestamp = splitted[3].split("_")
                item["Timestamp"] = int(datetime.strptime(temp_timestamp[0]+"_"+temp_timestamp[1], "%Y-%m-%d_%H-%M-%S").timestamp())
                item["Filename"] = splitted[3]
                item["LastModified"] = int(item["LastModified"].timestamp())
                item["ETag"] = item["ETag"].replace('"', '')
            else:
                if len(splitted) < 3:
                    print(item)
                    continue
                
                item["Sensor"] = splitted[0]
                item["Protocol"] = splitted[1]
                temp_timestamp = splitted[2].split("_")
                item["Timestamp"] = int(datetime.strptime(temp_timestamp[0]+"_"+temp_timestamp[1], "%Y-%m-%d_%H-%M-%S").timestamp())
                item["Filename"] = splitted[2]
                item["LastModified"] = int(item["LastModified"].timestamp())
                item["ETag"] = item["ETag"].replace('"', '')
            
            s3_cache.append(item)

    s3_cache_df = pd.DataFrame(s3_cache)
    s3_cache_df.to_csv("%s.csv" % bucket_name, index=False)

if __name__ == "__main__":
    fetch_s3_catalog("archive-orcasound-net")
