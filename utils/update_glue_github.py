from awsPy.aws_authorization import aws_connector
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
from pathlib import Path
import os, shutil, json, re
path = os.getcwd()
parent_path = str(Path(path).parent.parent)

def find_input_from_query(client, TableName,query):
    """
    """
    ### connect AWS
    glue = service_glue.connect_glue(client = client)
    list_input = []
    tables = glue.get_tables(full_output = False)
    regex_matches = re.findall(r'(?=\.).*?(?=\s)|(?=\.\").*?(?=\")', query)
    for i in regex_matches:
        cleaning = i.lstrip().rstrip().replace('.', '').replace('"', '')
        if cleaning in tables and cleaning != TableName:
            list_input.append(cleaning)

    return list(dict.fromkeys(list_input))

def automatic_update(
list_tables = 'automatic',
 automatic= True,
  new_schema = None,
   client = None,
   TableName = None,
   query = None):
    """
    list_tables-> tupple: table, database
    [
        ('journals_scimago', "scimago"),
        ('papers_meta_analysis',"esg"),
        ('papers_meta_analysis_new', "esg")]
]
    """
    if list_tables == 'automatic':
        list_tables = find_input_from_query(client = client, TableName = TableName,query = query)
    ### find database
    tables= glue.get_tables(full_output = True)
    list_to_search = []
    for table in list_tables:
        db = next((item for item in tables if item["Name"] == table), None)['DatabaseName']
        list_to_search.append((table, db))

    ### Fetch previous variables information
    comments = [
    glue.get_table_information(
    database = j,
    table = i)['Table']['StorageDescriptor']['Columns']
    for i, j in list_to_search
]

    comments = [item for sublist in comments for item in sublist]
    ### get schema new table
    schema = glue.get_table_information(
    database = DatabaseName,
    table = table_name)['Table']['StorageDescriptor']['Columns']
    ### Match known comments
    for name in schema:
        com = next((item for item in comments if item["Name"] == name['Name']), None)
        try:
            name['Comment'] = com['Comment']
        except:
            pass
    to_update = [i for i in schema if i['Comment'] == '']
    if automatic == True:
        for i in to_update:
            i['Comment'] = i['Name'].replace("_", " ")
        return schema
    if new_schema != None:
        for name in new_schema:
            com = next((i for i, item in enumerate(schema) if item["Name"] == name['Name']), None)
            if name['Comment'] != '':
                schema[com]['Comment'] = name['Comment']
            else:
                schema[com]['Comment'] = name['Name'].replace("_", " ")
        return schema
    else:
        return to_update

def update_glue_github(client, dic_information):
    """
    - DatabaseName:
    - TablePrefix:
    - input:
    - filename: Name of the notebook or Python script: to indicate
    - Task ID: from Coda
    - index_final_table: a list to indicate if the current table is used to prepare the final table(s). If more than one, pass the index. Start at 0
    - if_final: A boolean. Indicates if the current table is the final table -> the one the model will be used to be trained
    - schema: glue schema with comment
    - description: details query objective
    - query: query used to create table
    {
    'name_json':'parameters_ETL_TEMPLATE.json',
    'partition_keys':[""],
    'notebookname':"XX.ipynb",
    'index_final_table':[0],
    'if_final': 'False',
    'schema':schema,
    'description':description
    'query'
}
    """
    ### connect AWS
    glue = service_glue.connect_glue(client = client)
    ### update schema
    glue.update_schema_table(
    database = dic_information['DatabaseName'],
    table = dic_information['TableName'],
    schema= dic_information['schema'])

    ### update github
    path_json = os.path.join(str(Path(path).parent.parent), 'utils',dic_information['name_json'])
    with open(path_json) as json_file:
        parameters = json.load(json_file)

    github_url = os.path.join(
        "https://github.com/",
        parameters['GLOBAL']['GITHUB']['owner'],
        parameters['GLOBAL']['GITHUB']['repo_name'],
        "blob/master",
        re.sub(parameters['GLOBAL']['GITHUB']['repo_name'],
               '', re.sub(
                   r".*(?={})".format(parameters['GLOBAL']['GITHUB']['repo_name'])
                   , '', path))[1:],
        re.sub('.ipynb','.md',dic_information['notebookname'])
    )

    ###
    if dic_information['list_input_automatic']:
        list_input = []
        tables = glue.get_tables(full_output = False)
        regex_matches = re.findall(r'(?=\.).*?(?=\s)|(?=\.\").*?(?=\")', dic_information['query'])
        for i in regex_matches:
            cleaning = i.lstrip().rstrip().replace('.', '').replace('"', '')
            if cleaning in tables and cleaning != dic_information['TableName']:
                list_input.append(cleaning)
    else:
        list_input = dic_information['list_input_automatic']

    json_etl = {
        'description': dic_information['description'],
        'query': dic_information['query'],
        'schema': dic_information['schema'],
        'partition_keys': dic_information['partition_keys'],
        'metadata': {
            'DatabaseName': dic_information['DatabaseName'],
            'TableName': dic_information['TableName'],
            'input': list_input,
            'target_S3URI': os.path.join('s3://', dic_information['bucket'], dic_information['s3_output']),
            'from_athena': 'True',
            'filename': dic_information['notebookname'],
            'index_final_table' : dic_information['index_final_table'],
            'if_final': dic_information['if_final'],
             'github_url':github_url
        }
    }
    index_to_remove = next(
                    (
                        index
                        for (index, d) in enumerate(parameters['TABLES']['TRANSFORMATION']['STEPS'])
                        if d['metadata']['TableName'] == dic_information['TableName']
                    ),
                    None,
                )
    if index_to_remove != None:
        parameters['TABLES']['TRANSFORMATION']['STEPS'].pop(index_to_remove)
    parameters['TABLES']['TRANSFORMATION']['STEPS'].append(json_etl)
    print("Currently, the ETL has {} tables".format(len(parameters['TABLES']['TRANSFORMATION']['STEPS'])))
    with open(path_json, "w") as json_file:
        json.dump(parameters, json_file)
