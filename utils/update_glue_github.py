from awsPy.aws_authorization import aws_connector
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
import os
from pathlib import Path
path = os.getcwd()
parent_path = str(Path(path).parent.parent)

def update_glue_github(dic_information):
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
    name_credential= dic_information['name_credential']
    path_cred = "{0}/creds/{1}".format(parent_path, name_credential)
    region = dic_information['region']
    con = aws_connector.aws_instantiate(credential = path_cred,
                                           region = region)
    client= con.client_boto()
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
    list_input = []
    tables = glue.get_tables(full_output = False)
    regex_matches = re.findall(r'(?=\.).*?(?=\s)|(?=\.\").*?(?=\")', query)
    for i in regex_matches:
        cleaning = i.lstrip().rstrip().replace('.', '').replace('"', '')
        if cleaning in tables and cleaning != dic_information['TableName']:
            list_input.append(cleaning)

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
