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
    glue = service_glue.connect_glue(client = client)
    tables= glue.get_tables(full_output = True)
    list_to_search = []
    for table in list_tables:
        db = next((item for item in tables if item["Name"] == table), None)['DatabaseName']
        list_to_search.append((table, db))

    ### find db new table
    db_new = next((item for item in tables if item["Name"] == TableName), None)['DatabaseName']
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
    database = db_new,
    table = TableName)['Table']['StorageDescriptor']['Columns']
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
        return [i[0] for i in list_to_search], schema
    if new_schema != None:
        for name in new_schema:
            com = next((i for i, item in enumerate(schema) if item["Name"] == name['Name']), None)
            if name['Comment'] != '':
                schema[com]['Comment'] = name['Comment']
            else:
                schema[com]['Comment'] = name['Name'].replace("_", " ")
        return [i[0] for i in list_to_search],  schema
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
    json_etl = {
        'description': dic_information['description'],
        'query': dic_information['query'],
        'schema': dic_information['schema'],
        'partition_keys': dic_information['partition_keys'],
        'metadata': {
            'DatabaseName': dic_information['DatabaseName'],
            'TableName': dic_information['TableName'],
            'input': dic_information['list_input'],
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


def find_duplicates(client, bucket, name_json,partition_keys, TableName):
    """
    """
    s3 = service_s3.connect_S3(client = client,
                      bucket = bucket, verbose = True)
    path_json = os.path.join(str(Path(path).parent.parent), 'utils',name_json)
    with open(path_json) as json_file:
        parameters = json.load(json_file)

    glue = service_glue.connect_glue(client = client)
    tables= glue.get_tables(full_output = True)
    DatabaseName = next((item for item in tables if item["Name"] == TableName), None)['DatabaseName']

    ### COUNT DUPLICATES
    if len(partition_keys) > 0:
        groups = ' , '.join(partition_keys)

        query_duplicates = parameters["ANALYSIS"]['COUNT_DUPLICATES']['query'].format(
                                    DatabaseName,TableName,groups
                                    )
        dup = s3.run_query(
                                    query=query_duplicates,
                                    database=DatabaseName,
                                    s3_output="SQL_OUTPUT_ATHENA",
                                    filename="duplicates_{}".format(TableName))
    return display(dup)

def count_missing(client, name_json, TableName):
    #from datetime import date
    #today = date.today().strftime('%Y%M%d')
    path_json = os.path.join(str(Path(path).parent.parent), 'utils',name_json)
    with open(path_json) as json_file:
        parameters = json.load(json_file)

    s3 = service_s3.connect_S3(client = client,
                      bucket = bucket, verbose = True)

    glue = service_glue.connect_glue(client = client)
    tables= glue.get_tables(full_output = True)
    DatabaseName = next((item for item in tables if item["Name"] == TableName), None)['DatabaseName']
    table_top = parameters["ANALYSIS"]["COUNT_MISSING"]["top"]
    table_middle = ""
    table_bottom = parameters["ANALYSIS"]["COUNT_MISSING"]["bottom"].format(
        DatabaseName, TableName
    )

    for key, value in enumerate(schema["StorageDescriptor"]["Columns"]):
        if key == len(schema["StorageDescriptor"]["Columns"]) - 1:

            table_middle += "{} ".format(
                parameters["ANALYSIS"]["COUNT_MISSING"]["middle"].format(value["Name"])
            )
        else:
            table_middle += "{} ,".format(
                parameters["ANALYSIS"]["COUNT_MISSING"]["middle"].format(value["Name"])
            )
    query = table_top + table_middle + table_bottom
    output = s3.run_query(
        query=query,
        database=DatabaseName,
        s3_output="SQL_OUTPUT_ATHENA",
        filename="count_missing",  ## Add filename to print dataframe
        destination_key=None,  ### Add destination key if need to copy output
    )
    display(
        output.T.rename(columns={0: "total_missing"})
        .assign(total_missing_pct=lambda x: x["total_missing"] / x.iloc[0, 0])
        .sort_values(by=["total_missing"], ascending=False)
        .style.format("{0:,.2%}", subset=["total_missing_pct"])
        .bar(subset="total_missing_pct", color=["#d65f5f"])
    )
