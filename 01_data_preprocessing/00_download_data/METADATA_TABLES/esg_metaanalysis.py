import pandas as pd
import numpy as np
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
from awsPy.aws_authorization import aws_connector
from GoogleDrivePy.google_drive import connect_drive
from GoogleDrivePy.google_authorization import authorization_service
from pathlib import Path
import os
import re
import json
from tqdm import tqdm

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
PATH_S3 = "DATA/FINANCE/ESG/META_ANALYSIS_NEW"  # Copy destination in S3 without bucket and "/" at the end
# GCP
auth = authorization_service.get_authorization(
    #path_credential_gcp=os.path.join(parent_path, "creds", "service.json"),
    path_credential_drive=os.path.join(parent_path, "creds"),
    verbose=False,
    scope=['https://www.googleapis.com/auth/spreadsheets.readonly',
           "https://www.googleapis.com/auth/drive"]
)
gd_auth = auth.authorization_drive(path_secret=os.path.join(
    parent_path, "creds", "credentials.json"))
drive = connect_drive.drive_operations(gd_auth)

# DOWNLOAD SPREADSHEET TO temporary_local_data folder
FILENAME_SPREADSHEET = "METADATA_TABLES_COLLECTION"
spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)
sheetName = 'DATA'
var = (
    drive.upload_data_from_spreadsheet(
        sheetID=spreadsheet_id,
        sheetName=sheetName,
        to_dataframe=True)
        .apply(pd.to_numeric, errors='ignore')
        .assign(
        id = lambda x: x['id'].ffill().astype(int),
        n = lambda x: x['n'].apply(pd.to_numeric, errors='coerce'),
        sr = lambda x: x['sr'].apply(pd.to_numeric, errors='coerce'),
        beta = lambda x: x['beta'].apply(pd.to_numeric, errors='coerce'),
        critical_99 = lambda x: x['critical_99'].apply(pd.to_numeric, errors='coerce'),
        critical_95 = lambda x: x['critical_95'].apply(pd.to_numeric, errors='coerce'),
        critical_90 = lambda x: x['critical_90'].apply(pd.to_numeric, errors='coerce'),
        true_standard_error = lambda x: x['true_standard_error'].apply(pd.to_numeric, errors='coerce'),
        true_t_value = lambda x: x['true_t_value'].apply(pd.to_numeric, errors='coerce'),
        true_critical = lambda x: x['beta']/x['true_standard_error']
        #significant = lambda x: np.where(x['true_stars'].isin(['1_PERCENT','5_PERCENT','10_PERCENT']), "YES", "NO")
        )
        .loc[lambda x: ~x["table_refer"].isin([np.nan])]
        .drop(columns = [])
        .replace(',', '', regex=True)
        .replace('\"', ' ', regex=True)
        .replace('\n', ' ', regex=True)
        .replace('#N/A|#VALUE!|\?\?\?|#DIV/0!', np.nan, regex=True)
        .drop(columns = ['image_equation', 'ID2'])
)

# using true critical value
var.loc[var['true_critical'] >=var['critical_99'], 'true_stars'] = "1_PERCENT"
var.loc[(var['true_critical'] >=var['critical_95']) & (var['true_critical'] <var['critical_99']), 'true_stars'] = "5_PERCENT"
var.loc[(var['true_critical'] >=var['critical_90']) & (var['true_critical'] <var['critical_95']), 'true_stars'] = "10_PERCENT"

# using true t-value
var.loc[(var['true_t_value'] >=var['critical_99']) & (~var['true_stars'].isin(['1_PERCENT','5_PERCENT','10_PERCENT'])), 'true_stars'] = "1_PERCENT"
var.loc[(var['true_t_value'] >=var['critical_95']) & (var['true_t_value'] <var['critical_99']) & (~var['true_stars'].isin(['1_PERCENT','5_PERCENT','10_PERCENT'])), 'true_stars'] = "5_PERCENT"
var.loc[(var['true_t_value'] >=var['critical_90']) & (var['true_t_value'] <var['critical_95']) & (var['true_stars'].isin(['1_PERCENT','5_PERCENT','10_PERCENT'])), 'true_stars'] = "10_PERCENT"


var = var.assign(
significant = lambda x: np.where(x['true_stars'].isin(['1_PERCENT','5_PERCENT','10_PERCENT']), "YES", "NO")
)

#var.loc[lambda x:
#(x['to_remove'].isin(['TO_KEEP'])) & (x['to_check_final'].isin(['NO']))
#].loc[lambda x: x['true_standard_error'].isin([np.nan])][
#['id','beta', 'sr', 'true_standard_error', 'test_standard_error']]
# READ DATA
# SAVE LOCALLY
input_path = os.path.join(parent_path, "00_data_catalog",
                          "temporary_local_data",  FILENAME_SPREADSHEET + ".csv")


# preprocess data
var.dtypes
var.to_csv(input_path, index=False)
# SAVE S3
s3.upload_file(input_path, PATH_S3)
# os.remove(input_path)

# ADD SHCEMA
# ADD SHCEMA
#var_dtype = (
#    pd.DataFrame(var.dtypes, columns=['type']).assign(
#        type=lambda x: x['type'].astype('str'))
#)
#for i in range(0, len(var_dtype)):
#    row = var_dtype.iloc[i, :]
#    if row.values[0] not in ['string', "object"]:
#        print(row)
#        type = 'float'
#    else:
#        type = 'string'
#    print("{},".format({"Name": row.name, "Type": type, "Comment": ""}))

schema = [
{'Name': 'id', 'Type': 'string', 'Comment': 'paper ID'},
{'Name': 'incremental_id', 'Type': 'string', 'Comment': 'row id'},
{'Name': 'paper_name', 'Type': 'string', 'Comment': 'Paper name'},
{'Name': 'dependent', 'Type': 'string', 'Comment': 'dependent variable'},
{'Name': 'independent', 'Type': 'string', 'Comment': 'independent variables'},
{'Name': 'lag', 'Type': 'string', 'Comment': 'the table contains lag or not'},
{'Name': 'interaction_term', 'Type': 'string', 'Comment': 'the table contains interaction terms or not'},
{'Name': 'quadratic_term', 'Type': 'string', 'Comment': 'the table contains quadratic terms or not'},
{'Name': 'n', 'Type': 'float', 'Comment': 'number of observations'},
{'Name': 'r2', 'Type': 'float', 'Comment': 'R square'},
{'Name': 'beta', 'Type': 'float', 'Comment': 'Beta coefficient'},
{'Name': 'sign', 'Type': 'string', 'Comment': 'sign positive or negative'},
{'Name': 'star', 'Type': 'string', 'Comment': 'Level of significant. *, ** or ***'},
{'Name': 'sr', 'Type': 'float', 'Comment': 'standard error'},
{'Name': 'p_value', 'Type': 'float', 'Comment': 'p value'},
{'Name': 't_value', 'Type': 'float', 'Comment': 'student value'},
{'Name': 'image', 'Type': 'string', 'Comment': 'Link row data image'},
{'Name': 'table_refer', 'Type': 'string', 'Comment': 'table number in the paper'},
{'Name': 'model_name', 'Type': 'string', 'Comment': 'Model name from Susie'},
{'Name': 'updated_model_name', 'Type': 'string', 'Comment': 'Model name adjusted by Thomas'},
{'Name': 'adjusted_model_name', 'Type': 'string', 'Comment': 'Model name normalised'},
{'Name': 'doi', 'Type': 'string', 'Comment': 'DOI'},
{'Name': 'drive_url', 'Type': 'string', 'Comment': 'paper link in Google Drive'},
{'Name': 'test_standard_error', 'Type': 'string', 'Comment': 'check if sr really standard error by comparing beta divided by sr and critical values'},
{'Name': 'test_p_value', 'Type': 'string', 'Comment': 'check if sign and p value match'},
{'Name': 'test_t_value', 'Type': 'string', 'Comment': 'check if t critial value and sign match'},
{'Name': 'should_t_value', 'Type': 'string', 'Comment': 'use sr as t value when mistake'},
{'Name': 'true_standard_error', 'Type': 'float', 'Comment': 'reconstructed standard error'},
{'Name': 'true_t_value', 'Type': 'float', 'Comment': 'reconstructed t value'},
{'Name': 'true_stars', 'Type': 'string', 'Comment': 'reconstructed stars'},
{'Name': 'adjusted_dependent', 'Type': 'string', 'Comment': 'reorganise dependent variable into smaller groups'},
{'Name': 'adjusted_independent', 'Type': 'float', 'Comment': 'reorganise independent variable into smaller group'},
{'Name': 'adjusted_model', 'Type': 'string', 'Comment': 'reorganise model variable into smaller group'},
{'Name': 'significant', 'Type': 'string', 'Comment': 'is beta significant'},
{'Name': 'to_check_final', 'Type': 'string', 'Comment': 'Final check rows'},
{'Name': 'critical_99', 'Type': 'string', 'Comment': '99 t stat critical value calculated based on nb obserations'},
{'Name': 'critical_90', 'Type': 'string', 'Comment': '90 t stat critical value calculated based on nb obserations'},
{'Name': 'critical_95', 'Type': 'string', 'Comment': '95 t stat critical value calculated based on nb obserations'},
]

# ADD DESCRIPTION
description = 'upload new values for papers in Drive'

glue = service_glue.connect_glue(client=client)

target_S3URI = os.path.join("s3://",bucket, PATH_S3)
name_crawler = "crawl-industry-name"
Role = 'arn:aws:iam::468786073381:role/AWSGlueServiceRole-crawler-datalake'
DatabaseName = "esg"
TablePrefix = 'papers_'  # add "_" after prefix, ex: hello_


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
filename = 'esg_metaanalysis.py'
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
