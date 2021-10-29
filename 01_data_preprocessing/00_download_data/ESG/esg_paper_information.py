import numpy as np
from tqdm import tqdm
import json
import re
import os
from pathlib import Path
from GoogleDrivePy.google_authorization import authorization_service
from GoogleDrivePy.google_drive import connect_drive
from awsPy.aws_authorization import aws_connector
from awsPy.aws_glue import service_glue
from awsPy.aws_s3 import service_s3
import pandas as pd

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
# Copy destination in S3 without bucket and "/" at the end
PATH_S3 = "DATA/FINANCE/ESG/META_ANALYSIS"
# GCP
auth = authorization_service.get_authorization(
    #path_credential_gcp=os.path.join(parent_path, "creds", "service.json"),
    path_credential_drive=os.path.join(parent_path, "creds"),
    verbose=False,
    # scope = ['https://www.googleapis.com/auth/spreadsheets.readonly',
    # "https://www.googleapis.com/auth/drive"]
)
gd_auth = auth.authorization_drive(path_secret=os.path.join(
    parent_path, "creds", "credentials.json"))
drive = connect_drive.drive_operations(gd_auth)

# DOWNLOAD SPREADSHEET TO temporary_local_data folder
FILENAME_SPREADSHEET = "CSR Excel File Meta-Analysis - Version 4 -  01.02.2021"
spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)
sheetName = 'Feuil1'
var = (
    drive.upload_data_from_spreadsheet(
        sheetID=spreadsheet_id,
        sheetName=sheetName,
        to_dataframe=True)
    .replace('#N/A|#VALUE!|\?\?\?', np.nan, regex=True)
    .replace('', np.nan)
    .replace('\n', ' ', regex=True)
    .replace('\"', ' ', regex=True)
)
# Clean column name
var.columns = (var.columns
               .str.strip()
               .str.replace(' ', '_')
               .str.replace(r'\/|\(|\)|\?|\.|\:|\-', '', regex=True)
               .str.replace('__', '_').str.replace('\\n', '', regex=True)
               )

# READ DATA
#var.loc[lambda x: x['Nr'].isin(["44"])]
#var.to_excel('test.xlsx')
to_numeric = [
    # 'CNRS_Ranking',
    'ranking',
    # 'Sample_size_number_of_companies',
    'First_date_of_observations',
    'Last_date_of_observations',
    'Number_of_observations',
    'Level_of_significancy',
    'Standard_Error',
    'tstatistic_calculated_with_formula',
    'pvalue_calculated_with_formula',
    'Effect_Coeffient_Estimator_Beta',
    'Adjusted_coefficient_of_determination',
    "Publication_year",
    "Sample_size_number_of_companies"
]

for n in to_numeric:
    var[n] = var[n].apply(pd.to_numeric, errors='coerce')

# To int
var = var.assign(
    #ranking=lambda x: pd.to_numeric(x['ranking']),
    First_date_of_observations=lambda x: pd.to_numeric(
        x['First_date_of_observations']),
    Last_date_of_observations=lambda x: pd.to_numeric(
        x['Last_date_of_observations']),
    Number_of_observations=lambda x: pd.to_numeric(
        x['Number_of_observations']),
    Sample_size_number_of_companies=lambda x: pd.to_numeric(
        x['Sample_size_number_of_companies']),
)

# Remove coma
var = var.replace(',', '', regex=True)

var = (
    var.replace({
        #"CNRS_Ranking": {'1eg': '1', '1g': '1', '?': np.nan},
        'Study_focused_on_social_environmental_behaviour': {
            'Environmental Social': 'Environmental and Social',
            'Social and environmental': 'Environmental and Social',
        },
        'Type_of_data': {'Cross-sectional data': 'Cross-section', 'Cross-section data': 'Cross-section',
                         'Longitudinal study': 'Panel data', 'Longitudinal data': 'Panel data'},
        # 'Econometric_method': {'2sls': np.nan,
        #                       'AR': np.nan,
        #                       'Correlation Matrix': np.nan,
        #                       'Fixed effects': np.nan,
        #                       'GLS': np.nan,
        #                       'GMM': np.nan,
        #                       'IV': np.nan,
        #                       'Multiple Linear Regression': np.nan,
        #                       'Newey': np.nan,
        #                       'Non-linear': np.nan,
        #                       'Partial Least Squares': np.nan,
        #                       'Piecewise': np.nan,
        #                       'Pooled': np.nan,
        #                       'Random Effects': np.nan,
        #                       'Simple Linear Regression': np.nan,
        #                       'Simultaneous model': np.nan,
        #                       'Structural': np.nan,
        #                       'Tobit': np.nan,
        #                       },
        "Lagged_CSR_explanatory_variable": {
            'Yes 1 lag': 'Yes',
            'Yes 2 day lag': 'Yes',
            'Yes 2 lags': 'Yes',
            'Yes 4 lags': 'Yes',
        },
        "Regions_of_selected_firms": {
            'U.S.': 'US',
            'USA.': 'US',
            'USA': 'US',
            'United Kingdom': 'UK',
        }
    },
    )
    .assign(
        kyoto_db=lambda x: np.where(
            x['First_date_of_observations'] >= 1997,
            "Yes",
            'No'
        ),
        crisis_db=lambda x: np.where(
            x['First_date_of_observations'] >= 2009,
            "Yes",
            'No'
        )
    )
    .assign(
        kyoto_db=lambda x: np.where(
            x['First_date_of_observations'].isna(),
            np.nan,
            x['kyoto_db']
        ),
        crisis_db=lambda x: np.where(
            x['First_date_of_observations'].isna(),
            np.nan,
            x['crisis_db']
        )
    )
)

# SAVE LOCALLY
var.columns = var.columns.str.lower()
input_path = os.path.join(parent_path, "00_data_catalog",
                          "temporary_local_data",  FILENAME_SPREADSHEET + ".csv")
var.to_csv(input_path, index=False)

#for i in var.columns:
#    if i in to_numeric:
#        type = 'float'
#    else:
#        type = 'string'
#    print({"Name":i.lower(), "Type":type, "Comment":""})
# SAVE S3
#var.dtypes
s3.upload_file(input_path, PATH_S3)
os.remove(input_path)
# ADD SHCEMA
schema = [
    {'Name': 'nr', 'Type': 'string', 'Comment': ''},
{'Name': 'title', 'Type': 'string', 'Comment': ''},
{'Name': 'first_author', 'Type': 'string', 'Comment': ''},
{'Name': 'second_author', 'Type': 'string', 'Comment': ''},
{'Name': 'third_author', 'Type': 'string', 'Comment': ''},
{'Name': 'publication_year', 'Type': 'float', 'Comment': ''},
{'Name': 'publication_type', 'Type': 'string', 'Comment': ''},
{'Name': 'publication_name', 'Type': 'string', 'Comment': ''},
{'Name': 'cnrs_ranking', 'Type': 'string', 'Comment': ''},
{'Name': 'ranking', 'Type': 'float', 'Comment': ''},
{'Name': 'peer_reviewed', 'Type': 'string', 'Comment': ''},
{'Name': 'study_focused_on_social_environmental_behaviour', 'Type': 'string', 'Comment': ''},
{'Name': 'comments_on_sample', 'Type': 'string', 'Comment': ''},
{'Name': 'type_of_data', 'Type': 'string', 'Comment': ''},
{'Name': 'sample_size_number_of_companies', 'Type': 'float', 'Comment': ''},
{'Name': 'first_date_of_observations', 'Type': 'float', 'Comment': ''},
{'Name': 'last_date_of_observations', 'Type': 'float', 'Comment': ''},
{'Name': 'number_of_observations', 'Type': 'float', 'Comment': ''},
{'Name': 'regions_of_selected_firms', 'Type': 'string', 'Comment': ''},
{'Name': 'study_focusing_on_developing_or_developed_countries', 'Type': 'string', 'Comment': ''},
{'Name': 'measure_of_corporate_social_responsibility_crp', 'Type': 'string', 'Comment': ''},
{'Name': 'csr_7_categories', 'Type': 'string', 'Comment': ''},
{'Name': 'csr_20_categories', 'Type': 'string', 'Comment': ''},
{'Name': 'unit_for_measure_of_environmental_behaviour', 'Type': 'string', 'Comment': ''},
{'Name': 'measure_of_financial_performance', 'Type': 'string', 'Comment': ''},
{'Name': 'cfp_26_categories', 'Type': 'string', 'Comment': ''},
{'Name': 'unit_for_measure_of_financial_performance', 'Type': 'string', 'Comment': ''},
{'Name': 'cfp_4_categories', 'Type': 'string', 'Comment': ''},
{'Name': 'lagged_csr_explanatory_variable', 'Type': 'string', 'Comment': ''},
{'Name': 'evaluation_method_of_the_link_between_csr_and_cfp', 'Type': 'string', 'Comment': ''},
{'Name': 'developed_new', 'Type': 'string', 'Comment': ''},
{'Name': 'definition_of_cfp_as_dependent_variable', 'Type': 'string', 'Comment': ''},
{'Name': 'comments', 'Type': 'string', 'Comment': ''},
{'Name': 'cfp_regrouping', 'Type': 'string', 'Comment': ''},
{'Name': 'level_of_significancy', 'Type': 'float', 'Comment': ''},
{'Name': 'sign_of_effect', 'Type': 'string', 'Comment': ''},
{'Name': 'standard_error', 'Type': 'float', 'Comment': ''},
{'Name': 'tstatistic_calculated_with_formula', 'Type': 'float', 'Comment': ''},
{'Name': 'pvalue_calculated_with_formula', 'Type': 'float', 'Comment': ''},
{'Name': 'effect_coeffient_estimator_beta', 'Type': 'float', 'Comment': ''},
{'Name': 'adjusted_coefficient_of_determination', 'Type': 'float', 'Comment': ''},
{'Name': 'econometric_method', 'Type': 'string', 'Comment': ''},
{'Name': 'kyoto_db', 'Type': 'string', 'Comment': ''},
{'Name': 'crisis_db', 'Type': 'string', 'Comment': ''},
{'Name': 'row_id_excel', 'Type': 'string', 'Comment': 'link to original row'}
]

# ADD DESCRIPTION
description = 'Download papers information for meta analysis'

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
