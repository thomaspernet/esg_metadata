import json
import re, os
from pathlib import Path

from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS
from diagrams.aws.database import Redshift, RDS
from diagrams.aws.integration import SQS
from diagrams.aws.storage import S3
#path_json = "parameters_ETL_Financial_dependency_pollution.json"
#with open(path_json) as json_file:
#    data = json.load(json_file)

#create_schema(path_json = path_json)
#path_save_image = os.getcwd()

def generate_graph_etl(data, list_final, path_save_image):
    """
    """

    list_path = []
    for ind, list_table in enumerate(list_final):

        intend = '    '
        template_schema = "\n\n"
        list_index = []
        list_origin = []
        list_preparation = []
        list_tranformation = []
        list_input_transform = []

        index_name = "temp_final_{0}".format(ind)
        dic_index = {'table':list_table['final_table'], 'index':index_name, 'index_name':index_name}
        list_index.append(dic_index)

        LOCALPATH_RAW_DIAGRAM = os.path.join(path_save_image, "IMAGES", list_table['final_table'])
        data['GLOBAL']['GITHUB']['owner']
        PATH_FOR_GITHUB = os.path.join(
        "https://raw.githubusercontent.com/",
        data['GLOBAL']['GITHUB']['owner'],
        data['GLOBAL']['GITHUB']['repo_name'],"master", 'utils', 'IMAGES',
        list_table['final_table'] + '.jpg')

        diagram = """
from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS
from diagrams.aws.database import Redshift, RDS
from diagrams.aws.integration import SQS
from diagrams.aws.storage import S3

with Diagram("{0}", show=False, filename="{1}", outformat="jpg"):\n
""".format(list_table['final_table'].replace("_"," ").upper(), LOCALPATH_RAW_DIAGRAM)
        #template_creation = """
        #with Cluster("CREATION"):\n
        #"""
        #template_preparation = """
        #with Cluster("PREPARATION"):\n
        #"""
        #template_transformation = """
        #with Cluster("TRANSFORMATION"):\n
        #"""
        template_final = """
     with Cluster("FINAL"):\n
"""
        add_origin = False
        add_preparation = False
        add_transformation= False
        for i, val in enumerate(sorted(list_table['pipeline'], key = lambda i: i['origin'], reverse = False)):
            if len([t['index_name'] for t in list_index if t['table'] == val['output']]) ==0:
                index_name = "temp_{0}".format(i +1)
                dic_index = {'table':val['output'], 'index':i, 'index_name':index_name}
                list_index.append(dic_index)

        for i, val in enumerate(sorted(list_table['pipeline'], key = lambda i: i['origin'], reverse = False)):
            if val['origin'] == 'CREATION':
                add_origin = True
                table  = "'{}'".format(val['output'])
                index_name = [t['index_name'] for t in list_index if t['table'] == val['output']]
                index_final = [t['index_name'] for t in list_index if t['table'] == list_table['final_table']]
                var = '{0} {1} = S3({2})\n'.format(intend , index_name[0], table)
                if var not in list_origin:
                    list_origin.append(var)
                #template_creation += var
                ### reverse the position of the table to make it nicer
                template_schema += "{0} {2} << {1}\n".format(intend,index_name[0], index_final[0])

            if val['origin'] == 'PREPARATION':
                add_preparation = True
                table  = "'{}'".format(val['output'])
                index_name = [t['index_name'] for t in list_index if t['table'] == val['output']]
                var = '{0} {1} = ECS({2})\n'.format(intend , index_name[0], table)
                if var not in list_preparation:
                    list_preparation.append(var)
                #template_preparation += var
            if val['origin'] == 'TRANSFORMATION':
                temp = val
                add_transformation = True
                table  = "'{}'".format(val['output'])
                index_name = [t['index_name'] for t in list_index if t['table'] == val['output']]
                index_final = [t['index_name'] for t in list_index if t['table'] == list_table['final_table']]
                var = '{0} {1} = SQS({2})\n'.format(intend , index_name[0], table)
                if var not in list_tranformation:
                    list_tranformation.append(var)
                #template_transformation += var
                ### connection
                connection = sort_origin(data, val['input'], reverse =False)
                template_schema_transform = ""
                size_connection = len(connection)
                for ind, tab in enumerate(connection):
                    index = [t['index_name'] for t in list_index if t['table'] == tab['table']]
                    if len(index) > 0:
                        template = "{0} >>".format(index[0])
                        template_schema_transform += template
                        template_schema_transform += "{} >> {}\n".format(index_name[0], index_final[0])
                        add_intend = True
                    else:
                        ## Add creation table to transform without PREPARATION
                        var = '{0} input_{1} = S3("{2}")\n'.format(intend , tab['table'], tab['table'])
                        if var not in list_origin:
                            list_origin.append(var)
                        input_temp = "input_{0}".format(tab['table'])
                        #template_schema_transform += template
                        if input_temp + index_name[0] not in list_input_transform:
                            if ind == size_connection-1:
                                template_schema_transform += "{} {} >> {} >> {}\n".format(intend, input_temp, index_name[0], index_final[0])
                            else:
                                ### the last input will link the TRANSFORMATION to final table
                                ### it avoids having many lines
                                template_schema_transform += "{} {} >> {}\n".format(intend, input_temp, index_name[0])

                            list_input_transform.append(input_temp +  index_name[0])
                        add_intend = False

                if add_intend:
                    template_schema += "{} {}".format(intend, template_schema_transform)
                else:
                    template_schema += "{}".format(template_schema_transform)
        table  = "'{}'".format(list_table['final_table'])
        var = '{0} {1} = Redshift({2})\n'.format(intend*2, index_final[0], table)
        template_final += var

        if add_origin:
        #    template_creation += ''.join(list_origin)
        #    diagram += template_creation
            diagram += ''.join(list_origin)
        if add_preparation:
            diagram += ''.join(list_preparation)
        #    template_preparation += ''.join(list_preparation)
        #    diagram += template_preparation
        if add_transformation:
            diagram += ''.join(list_tranformation)
        #    template_transformation += ''.join(list_tranformation)
        #    diagram += template_transformation

        diagram += template_final + template_schema

        if os.path.isdir(os.path.join(path_save_image, "IMAGES")) == False:
            os.mkdir(os.path.join(path_save_image, "IMAGES"))
        script_filename = os.path.join(path_save_image, "IMAGES","script_diagram_{}.py".format(list_table['final_table']))
        text_file = open(script_filename, "w")
        text_file.write(diagram)
        text_file.close()
        os.system("python '{}'".format(script_filename))
        list_path.append(PATH_FOR_GITHUB)
    return list_path


# Construct data lineage
## Final table known
def find_input(data, output):
    """
    """
    input = [val['metadata']['input'] for i, val in enumerate(data['TABLES']['TRANSFORMATION']['STEPS']) if
    val['metadata']['TableName'] == output]
    origin = 'TRANSFORMATION'
    if len(input) == 0:
        input = [val['metadata']['input'] for i, val in enumerate(data['TABLES']['PREPARATION']['STEPS']) if
        val['metadata']['TableName'] == output]
        origin = 'PREPARATION'
    #if len(input) == 0:
    #    input = [[val['metadata']['TableName'] for i, val in enumerate(data['TABLES']['CREATION']['ALL_SCHEMA']) if
    #    val['metadata']['TableName'] == output]]
    if len(input)> 0:
        input = input[0]
    else:
        origin = 'CREATION'
        input = [None]
    return origin, input

def sort_origin(data, list_to_check, reverse =True):
    """
    """
    list_temp  = []
    for t in list_to_check:
        origin, test = find_input(data = data,output = t)
        dic_temp = {'table':t, 'origin':origin}
        list_temp.append(dic_temp)
    list_input = sorted(list_temp, key = lambda i: i['origin'], reverse = reverse)
    return list_input

def find_github_url(data, table_name):
    """
    """
    info = [val for i, val in enumerate(data['TABLES']['TRANSFORMATION']['STEPS']) if
    val['metadata']['TableName'] == table_name]
    #url = info[0]['metadata']['github_url']
    #desc = info[0]['description']
    if len(info) == 0:
        info = [val for i, val in enumerate(data['TABLES']['PREPARATION']['STEPS']) if
        val['metadata']['TableName'] == table_name]
        #url = info[0]['metadata']['github_url']
        #desc = info[0]['description']
    if len(info) == 0:
        info = [val for i, val in enumerate(data['TABLES']['CREATION']['ALL_SCHEMA']) if
        val['metadata']['TableName'] == table_name]
        #url = info[0]['metadata']['github_url']
        #desc = info[0]['description']

    if len(info) > 0:
        url = info[0]['metadata']['github_url']
        desc = info[0]['description']
    else:
        url  = None
        desc = None
    return url, desc

#find_github_url('china_credit_constraint')

def organise_table_md(data, dic_final, output):
    """
    """
    ### Create md
    md = ["# {}\n".format(dic_final['final_table']).replace("_", " ").upper()]
    desc_final = [val['description'] for i, val in enumerate(data['TABLES']['TRANSFORMATION']['STEPS']) if
    val['metadata']['TableName'] == output]
    url, desc = find_github_url(data = data,table_name = dic_final['final_table'])
    md.append("{}\n".format(desc_final[0]))
    md.append("* **[{0}]({1})**: {2}\n".format(
    dic_final['final_table'],
    url,
    desc
    )
    )

    md_creation = ["    * CREATION\n"]
    indent_0 = '    '
    #indent_1 = indent_1 * (2)
    #indent_2 = indent_1 * (4)
    for i in dic_final['pipeline']:
        if i['origin'] == 'TRANSFORMATION':
            transform = i
            url, desc = find_github_url(data = data,table_name = transform['output'])
            md.append("{}* TRANSFORMATION\n".format(indent_0))
            md.append("{0}* [{1}]({2}): {3}\n".format(
            indent_0 * 2,
            transform['output'],
            url,
            desc
            ))
            nb_input = len(transform['input'])
            nb = 0
            ## Sort ORIGIN
            list_input = sort_origin(data = data,list_to_check= transform['input'])
            while nb < nb_input:
                url, desc = find_github_url(data = data,table_name = list_input[nb]['table'])
                md.append("{0}* {1}\n".format(indent_0 * 3, list_input[nb]['origin']))
                md.append("{0}* [{1}]({2}): {3}\n".format(
                indent_0 * 4,
                list_input[nb]['table'],
                url,
                desc
                ))
                ### Find input

                list_input_1 = [val['input'] for i, val in enumerate(dic_final['pipeline']) if
                val['output'] == list_input[nb]['table']]
                nb_1 = 0
                nb += 1
                if len(list_input_1) > 0:
                    list_input_1 = list_input_1[0]

                    list_input_1 = sort_origin(data = data,list_to_check= list_input_1)
                    nb_input_1 = len(list_input_1)

                    while nb_1 < nb_input_1:

                        if list_input_1[nb_1]['table'] !=  None:
                            url, desc =find_github_url(data = data,table_name = list_input_1[nb_1]['table'])
                            md.append("{0}* {1}\n".format(indent_0 * 5, list_input_1[nb_1]['origin']))
                            md.append("{0}* [{1}]({2}): {3}\n".format(
                            indent_0 * 6,
                            list_input_1[nb_1]['table'],
                            url,
                            desc
                            ))
                        nb_1 +=1

        if i['origin'] == 'CREATION':
            url, desc = find_github_url(data = data,table_name = i['output'])
            md_creation.append("{0}* [{1}]({2}): {3}\n".format(
            indent_0 * 2,
            i['output'],
            url,
            desc
            ))

    md.extend(md_creation)
    return md

def create_schema(path_json, path_save_image):
    with open(path_json) as json_file:
        data = json.load(json_file)

    list_final = []
    list_README = []
    for i, val in enumerate(data['TABLES']['TRANSFORMATION']['STEPS']):
        if val['metadata']['if_final'] == 'True':

            output = val['metadata']['TableName']
            list_input = val['metadata']['input']
            list_input = list(dict.fromkeys(list_input))
            ### order table TRANSFORMATION, PREPARATION, CREATION
            list_input = sort_origin(data = data,list_to_check= list_input)
            ### find all inputs
            ### Search in TRANSFORMATION
            dic_final = {'final_table': output}
            dic_final['pipeline'] = []
            for t in list_input:

                ## check if table already yin pipeline
                exist = [tab for tab in dic_final['pipeline'] if tab['output'] == t['table']]
                if len(exist) ==0:
                    list_output_input = list()
                    dic_ = {'output':t['table']}
                    origin, test = find_input(data = data,output = t['table'])
                    dic_['origin'] = origin
                    dic_['input'] = test
                    list_output_input.append(dic_)

                    while len(test) >0:
                        i = 0
                        dic_ = {'output':test[i]}
                        origin,input = find_input(data = data,output = test[i])
                        dic_['origin'] = origin
                        dic_['input'] = input
                        #list_output_input.append(dic_)
                        if input[0] != None:
                            list_output_input.append(dic_)
                            j = 0
                            while len(input) >0:
                                dic_ = {'output':input[j]}
                                origin, input = find_input(data = data,output = input[j])
                                dic_['origin'] = origin
                                dic_['input'] = input
                                if input[0] != None:
                                    list_output_input.append(dic_)
                                j=+1
                                input = input[j:]
                        i=+1
                        test = test[i:]
                    dic_final['pipeline'].extend(list_output_input)
            list_final.append(dic_final)
            README = organise_table_md(data = data,dic_final = dic_final, output = output)
            list_README.append(README)

    path_to_add = generate_graph_etl(data = data,list_final =list_final, path_save_image = path_save_image)
    ### append to readme
    add_image = """
### ETL diagrams

"""
    for i, val in enumerate(path_to_add):
        #i.split("/")[-1]
        path = "{0}\n\n![]({1})\n\n".format(add_image, val)
        list_README[i].append(path)

    README = ""
    for line in list_README:
        for lines in line:
            README += lines

    with open('README.md', "w") as outfile:
        outfile.write(README)
