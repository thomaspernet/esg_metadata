import pandas as pd
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
from awsPy.aws_authorization import aws_connector
from pathlib import Path
import os
import re
import json
from tqdm import tqdm
import requests

# Connect to Cloud providers
path = os.getcwd()
parent_path = str(Path(path).parent.parent.parent)
name_credential = 'financial_dep_SO2_accessKeys.csv'
region = 'eu-west-2'
bucket = 'datalake-london'
path_cred = "{0}/creds/{1}".format(parent_path, name_credential)

# AWS
con = aws_connector.aws_instantiate(credential=path_cred,
                                    region=region)
client = con.client_boto()
s3 = service_s3.connect_S3(client=client,
                           bucket=bucket, verbose=True)
PATH_S3 = "DATA/JOURNALS/SCIMAGO"  # Copy destination in S3 without bucket and "/" at the end
# GCP

# DOWNLOAD DATA TO temporary_local_data folder
url = 'https://www.scimagojr.com/journalrank.php?out=xls'
r = requests.get(url, allow_redirects=True)

open('journals_scimago.csv', 'wb').write(r.content)

var = (
pd.read_csv('journals_scimago.csv',sep=';', low_memory=False)
.replace(',', '.', regex=True)
.apply(pd.to_numeric, errors='ignore')
)
var.columns = (var.columns
               .str.strip()
               .str.replace(' ', '_', regex = True)
               .str.replace('\.', '', regex = True)
               .str.replace('_/_', '_', regex = True)
               .str.replace('\(|\)', '', regex = True)
               .str.lower()
               )

# READ DATA
input_path = os.path.join(parent_path, "00_data_catalog",
                          "temporary_local_data", "journals_scimago" + ".csv")

var.to_csv(input_path, index=False)
# SAVE S3
s3.upload_file(input_path, PATH_S3)
os.remove("journals_scimago.csv")
os.remove(input_path)

# ADD SHCEMA
# ADD SHCEMA
var_dtype = (
    pd.DataFrame(var.dtypes, columns=['type']).assign(
        type=lambda x: x['type'].astype('str'))
)
for i in range(0, len(var_dtype)):
    row = var_dtype.iloc[i, :]
    if row.values[0] not in ['string', "object"]:
        type = 'float'
    else:
        type = 'string'
    print("{},".format({"Name": row.name, "Type": type, "Comment": ""}))

schema = [
{'Name': 'rank', 'Type': 'bigint', 'Comment': 'journal rank'},
{'Name': 'sourceid', 'Type': 'bigint', 'Comment': 'source'},
{'Name': 'title', 'Type': 'string', 'Comment': 'Journal title'},
{'Name': 'type', 'Type': 'string', 'Comment': 'Type of journal'},
{'Name': 'issn', 'Type': 'string', 'Comment': 'ISSN'},
{'Name': 'sjr', 'Type': 'float', 'Comment': 'SCImago Journal Rank'},
{'Name': 'sjr_best_quartile', 'Type': 'string', 'Comment': 'SCImago Journal Rank rank by quartile'},
{'Name': 'h_index', 'Type': 'float', 'Comment': 'h-index is an author-level metric that measures both the productivity and citation impact of the publications of a scientist or scholar'},
{'Name': 'total_docs', 'Type': 'float', 'Comment': 'total doc in 2020'},
{'Name': 'total_docs_3years', 'Type': 'float', 'Comment': 'total doc past 3 years'},
{'Name': 'total_refs', 'Type': 'float', 'Comment': 'total references'},
{'Name': 'total_cites_3years', 'Type': 'float', 'Comment': 'total cited last 3 years'},
{'Name': 'citable_docs_3years', 'Type': 'float', 'Comment': 'citable doc'},
{'Name': 'cites_doc_2years', 'Type': 'float', 'Comment': 'citation per doc over the last 3 years'},
{'Name': 'ref_doc', 'Type': 'float', 'Comment': 'number of reference per doc'},
{'Name': 'country', 'Type': 'string', 'Comment': 'country of origin'},
{'Name': 'region', 'Type': 'string', 'Comment': 'region of origin'},
{'Name': 'publisher', 'Type': 'string', 'Comment': 'publisher'},
{'Name': 'coverage', 'Type': 'string', 'Comment': 'coverage'},
{'Name': 'categories', 'Type': 'string', 'Comment': 'categories'}
]

# ADD DESCRIPTION
description = 'SCImago journal database'

glue = service_glue.connect_glue(client=client)

target_S3URI = os.path.join("s3://",bucket, PATH_S3)
name_crawler = "crawl-industry-name"
Role = 'arn:aws:iam::468786073381:role/AWSGlueServiceRole-crawler-datalake'
DatabaseName = "scimago"
TablePrefix = 'journals_'  # add "_" after prefix, ex: hello_


glue.create_table_glue(
    target_S3URI,
    name_crawler,
    Role,
    DatabaseName,
    TablePrefix,
    from_athena=False,
    update_schema=schema,
)

# Add tp ETL parameter files
filename = 'scimagojr.py'
path_to_etl = os.path.join(
    str(Path(path).parent.parent.parent), 'utils', 'parameters_ETL_esg_metadata.json')
with open(path_to_etl) as json_file:
    parameters = json.load(json_file)
github_url = os.path.join(
    "https://github.com/",
    parameters['GLOBAL']['GITHUB']['owner'],
    parameters['GLOBAL']['GITHUB']['repo_name'],
    re.sub(parameters['GLOBAL']['GITHUB']['repo_name'],
           '', re.sub(
               r".*(?={})".format(parameters['GLOBAL']['GITHUB']['repo_name']), '', path))[1:],
    filename
)
table_name = '{}{}'.format(TablePrefix, os.path.basename(target_S3URI).lower())
json_etl = {
    'description': description,
    'schema': schema,
    'partition_keys': [],
    'metadata': {
        'DatabaseName': DatabaseName,
        'TablePrefix': TablePrefix,
        'TableName': table_name,
        'target_S3URI': target_S3URI,
        'from_athena': 'False',
        'filename': filename,
        'github_url': github_url
    }
}


with open(path_to_etl) as json_file:
    parameters = json.load(json_file)

# parameters['TABLES']['CREATION']['ALL_SCHEMA'].pop(0)

index_to_remove = next(
    (
        index
        for (index, d) in enumerate(parameters['TABLES']['CREATION']['ALL_SCHEMA'])
        if d['metadata']['filename'] == filename
    ),
    None,
)
if index_to_remove != None:
    parameters['TABLES']['CREATION']['ALL_SCHEMA'].pop(index_to_remove)

parameters['TABLES']['CREATION']['ALL_SCHEMA'].append(json_etl)

with open(path_to_etl, "w")as outfile:
    json.dump(parameters, outfile)
