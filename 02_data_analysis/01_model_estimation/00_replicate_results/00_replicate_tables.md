---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernel_info:
    name: python3
  kernelspec:
    display_name: SoS
    language: sos
    name: sos
---

<!-- #region kernel="SoS" -->
# US Name
Estimate sign_of_effect as a function of  publication_year and others variables


# Description

Estimate multi-probit

## Variables
### Target

sign_of_effect

### Features

- publication_year
- publication_type
- cnrs_ranking
- study_focused_on_social_environmental_behaviour
- type_of_data
- number_of_observations
- evaluation_method_of_the_link_between_csr_and_cfp
- developed_new
- econometric_method

## Complementary information



# Metadata

- Key: 176_esg_metadata
- Epic: Models
- US: Replicate previous results
- Task tag: #data-analysis, #meta-analysis, #mutlinomial-probit
- Analytics reports: 

# Input Cloud Storage

## Table/file

**Name**

- https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/00_download_data/ESG/esg_metaanalysis.py

**Github**

- papers_meta_analysis


<!-- #endregion -->

<!-- #region kernel="SoS" -->
# Connexion server
<!-- #endregion -->

```sos kernel="SoS"
from awsPy.aws_authorization import aws_connector
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
from pathlib import Path
import pandas as pd
import numpy as np
#import seaborn as sns
import os, shutil, json
import sys

path = os.getcwd()
parent_path = str(Path(path).parent.parent.parent)


name_credential = 'financial_dep_SO2_accessKeys.csv'
region = 'eu-west-3'
bucket = 'datalake-datascience'
path_cred = "{0}/creds/{1}".format(parent_path, name_credential)
```

```sos kernel="SoS"
con = aws_connector.aws_instantiate(credential = path_cred,
                                       region = region)
client= con.client_boto()
s3 = service_s3.connect_S3(client = client,
                      bucket = bucket, verbose = False)
glue = service_glue.connect_glue(client = client) 
```

```sos kernel="SoS"
pandas_setting = True
if pandas_setting:
    #cm = sns.light_palette("green", as_cmap=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
```

```sos kernel="SoS" nteract={"transient": {"deleting": false}}
os.environ['KMP_DUPLICATE_LIB_OK']='True'

```

<!-- #region kernel="SoS" -->
# Load tables

Since we load the data as a Pandas DataFrame, we want to pass the `dtypes`. We load the schema from Glue to guess the types
<!-- #endregion -->

```sos kernel="SoS"
db = 'esg'
table = 'papers_meta_analysis'
```

```sos kernel="SoS"
dtypes = {}
schema = (glue.get_table_information(database = db,
                           table = table)
          ['Table']['StorageDescriptor']['Columns']
         )
for key, value in enumerate(schema):
    if value['Type'] in ['varchar(12)',
                         'varchar(3)',
                        'varchar(14)', 'varchar(11)']:
        format_ = 'string'
    elif value['Type'] in ['decimal(21,5)', 'double', 'bigint', 'int', 'float']:
        format_ = 'float'
    else:
        format_ = value['Type'] 
    dtypes.update(
        {value['Name']:format_}
    )
```

```sos kernel="SoS"
download_data = True
filename = 'df_{}'.format(table)
full_path_filename = 'SQL_OUTPUT_ATHENA/CSV/{}.csv'.format(filename)
path_local = os.path.join(str(Path(path).parent.parent.parent), 
                              "00_data_catalog/temporary_local_data")
df_path = os.path.join(path_local, filename + '.csv')
if download_data:
    
    s3 = service_s3.connect_S3(client = client,
                          bucket = bucket, verbose = False)
    query = """
    SELECT * 
    FROM {}.{}
    """.format(db, table)
    try:
        df = (s3.run_query(
            query=query,
            database=db,
            s3_output='SQL_OUTPUT_ATHENA',
            filename=filename,  # Add filename to print dataframe
            destination_key='SQL_OUTPUT_ATHENA/CSV',  #Use it temporarily
            dtype = dtypes
        )
                )
    except:
        pass
    s3.download_file(
        key = full_path_filename
    )
    shutil.move(
        filename + '.csv',
        os.path.join(path_local, filename + '.csv')
    )
    s3.remove_file(full_path_filename)
    df.head()
```

```sos kernel="SoS" nteract={"transient": {"deleting": false}}
pd.DataFrame(schema)
```

<!-- #region kernel="SoS" nteract={"transient": {"deleting": false}} -->
## Schema Latex table

To rename a variable, please use the following template:

```
{
    'old':'XX',
    'new':'XX_1'
    }
```

if you need to pass a latex format with `\`, you need to duplicate it for instance, `\text` becomes `\\text:

```
{
    'old':'working\_capital\_i',
    'new':'\\text{working capital}_i'
    }
```

Then add it to the key `to_rename`
<!-- #endregion -->

```sos kernel="SoS" nteract={"transient": {"deleting": false}}
add_to_dic = True
if add_to_dic:
    if os.path.exists("schema_table.json"):
        os.remove("schema_table.json")
    data = {'to_rename':[], 'to_remove':[]}
    dic_rename = [
        {
        'old':'working\_capital\_i',
        'new':'\\text{working capital}_i'
        },
        {
        'old':'periodTRUE',
        'new':'\\text{period}'
        },
        {
        'old':'tso2\_mandate\_c',
        'new':'\\text{policy mandate}_'
        },
    ]

    data['to_rename'].extend(dic_rename)
    with open('schema_table.json', 'w') as outfile:
        json.dump(data, outfile)
```

```sos kernel="SoS"
sys.path.append(os.path.join(parent_path, 'utils'))
import latex.latex_beautify as lb
#%load_ext autoreload
#%autoreload 2
```

```sos kernel="R"
options(warn=-1)
library(tidyverse)
#library(lfe)
#library(lazyeval)
library(nnet)
library('progress')
path = "../../../utils/latex/table_golatex.R"
source(path)
```

```sos kernel="R"
#library(mlogit)
```

```sos kernel="R"
%get df_path
df_final <- read_csv(df_path) %>%
mutate_if(is.character, as.factor) %>%
mutate(sign_of_effect = relevel(sign_of_effect, ref='Positive'),
      nr = as.factor(nr))
   # mutate_at(vars(starts_with("fe")), as.factor) %>%
#mutate(VAR_TO_RELEVEL = relevel(XX, ref='XX'))
```

<!-- #region kernel="SoS" -->
## Table 1:Multinomial logit

$$
\begin{aligned}
\text{Write your equation}
\end{aligned}
$$

**Candidates**

- publication_year
- ~publication_type~
- ~cnrs_ranking~
- study_focused_on_social_environmental_behaviour
- ~type_of_data~
- ~number_of_observations~
- evaluation_method_of_the_link_between_csr_and_cfp
- developed_new
- ~econometric_method~

More about multinomial logit:

- https://dynalist.io/d/BP0wtNmNu-p8iF_bzH0OKX6i
- https://www.princeton.edu/~otorres/LogitR101.pdf
<!-- #endregion -->

```sos kernel="R"
df_final%>%head(2)
```

```sos kernel="SoS" nteract={"transient": {"deleting": false}}
folder = 'Tables_0'
table_nb = 1
table = 'table_{}'.format(table_nb)
path = os.path.join(folder, table + '.txt')
if os.path.exists(folder) == False:
        os.mkdir(folder)
for ext in ['.txt', '.tex', '.pdf']:
    x = [a for a in os.listdir(folder) if a.endswith(ext)]
    [os.remove(os.path.join(folder, i)) for i in x]
```

```sos kernel="R"
table(df_final$sign_of_effect)
```

```sos kernel="R"
df_final %>% 
group_by(sign_of_effect) %>%
summarize(
    mean(publication_year), median(publication_year),
mean(number_of_observations, na.rm=TRUE), median(number_of_observations, na.rm=TRUE))
```

<!-- #region kernel="R" -->
Not enough variability in the outcome
<!-- #endregion -->

```sos kernel="R"
table(df_final$sign_of_effect, df_final$publication_type)
```

```sos kernel="R"
table(df_final$sign_of_effect, df_final$cnrs_ranking)
```

<!-- #region kernel="R" -->
Biased toward ` Environmental Social and Governance`
<!-- #endregion -->

```sos kernel="R"
table(df_final$sign_of_effect, df_final$study_focused_on_social_environmental_behaviour)
```

<!-- #region kernel="R" -->
Not enough variability in the outcome: `???` and `Longitudinal study`
<!-- #endregion -->

```sos kernel="R"
table(df_final$sign_of_effect, df_final$type_of_data)
```

```sos kernel="R"
table(df_final$sign_of_effect, df_final$evaluation_method_of_the_link_between_csr_and_cfp)
```

```sos kernel="R"
table(df_final$sign_of_effect, df_final$developed_new)
```

<!-- #region kernel="R" -->
Not enough variability in the outcome
<!-- #endregion -->

```sos kernel="R"
table(df_final$sign_of_effect, df_final$econometric_method)
```

<!-- #region kernel="R" -->
The model controls for the paper's title but is not reported in the table
<!-- #endregion -->

```sos kernel="R"
%get path table
t_0 <- multinom(sign_of_effect ~ 
                #number_of_observations + 
                publication_year + 
                #publication_type +
                study_focused_on_social_environmental_behaviour +
                evaluation_method_of_the_link_between_csr_and_cfp +
                developed_new + 
                nr,
                data=df_final)
            
dep <- "Dependent variable: Sign of effect"
#fe1 <- list(
#    c("XXXXX", "Yes")
#             )

#table_1 <- go_latex(list(
#    t_0#,t_1, t_2, t_3
#),
#    title="TITLE",
#    dep_var = dep,
#    addFE=FALSE,
#    save=TRUE,
#    note = FALSE,
#    name=path
#) 
stargazer(t_0, type="text",  omit = "nr", style = "qje")
```

<!-- #region kernel="R" -->
### relative risk ratios (marginal effect)

**How to read**

- Publication year
    - Keeping all other variables constant, if the publication increases one unit, the paper is 0.995 times more likely to stay in the negative category as compared to the positive category (the risk or odds is .05% lower).
    
When the RRR is above one, the odd of the positive class is stronger (larger probability, keeping everything else constant)    
<!-- #endregion -->

```sos kernel="R"
t_0.rrr = exp(coef(t_0))
stargazer(t_0,
          type="text",
          coef=list(t_0.rrr),
          p.auto=FALSE,
          omit = "nr")
```

```sos kernel="SoS"
#tbe1  = "Reference group: Positive " \
#"\sym{*} Significance at the 10\%, \sym{**} Significance at the 5\%, \sym{***} Significance at the 1\%."

#multicolumn ={
#    'Eligible': 2,
#    'Non-Eligible': 1,
#    'All': 1,
#    'All benchmark': 1,
#}

#multi_lines_dep = '(city/product/trade regime/year)'
#new_r = ['& Insignificant', 'Negative']
#lb.beautify(table_number = table_nb,
            #reorder_var = reorder,
            #multi_lines_dep = multi_lines_dep,
#            new_row= new_r,
            #multicolumn = multicolumn,
#            table_nte = tbe1,
#            jupyter_preview = True,
#            resolution = 200,
#            folder = folder)
```

<!-- #region kernel="SoS" nteract={"transient": {"deleting": false}} -->
# Generate reports
<!-- #endregion -->

```sos kernel="python3" nteract={"transient": {"deleting": false}} outputExpanded=false
import os, time, shutil, urllib, ipykernel, json
from pathlib import Path
from notebook import notebookapp
import sys
path = os.getcwd()
parent_path = str(Path(path).parent.parent.parent)
sys.path.append(os.path.join(parent_path, 'utils'))
import make_toc
```

```sos kernel="SoS"
name_json = 'parameters_ETL_pollution_credit_constraint.json'
path_json = os.path.join(str(Path(path).parent.parent), 'utils',name_json)
```

```sos kernel="python3" nteract={"transient": {"deleting": false}} outputExpanded=false
def create_report(extension = "html", keep_code = False, notebookname = None):
    """
    Create a report from the current notebook and save it in the 
    Report folder (Parent-> child directory)
    
    1. Exctract the current notbook name
    2. Convert the Notebook 
    3. Move the newly created report
    
    Args:
    extension: string. Can be "html", "pdf", "md"
    
    
    """
    
    ### Get notebook name
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[0].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+ \
                                             'api/sessions?token=' + \
                                             srv['token'])
            sessions = json.load(req)
            notebookname = sessions[0]['name']
        except:
            notebookname = notebookname  
    
    sep = '.'
    path = os.getcwd()
    #parent_path = str(Path(path).parent)
    
    ### Path report
    #path_report = "{}/Reports".format(parent_path)
    #path_report = "{}/Reports".format(path)
    
    ### Path destination
    name_no_extension = notebookname.split(sep, 1)[0]
    source_to_move = name_no_extension +'.{}'.format(extension)
    dest = os.path.join(path,'Reports', source_to_move)
    
    ### Generate notebook
    if keep_code:
        os.system('jupyter nbconvert --to {} {}'.format(
    extension,notebookname))
    else:
        os.system('jupyter nbconvert --no-input --to {} {}'.format(
    extension,notebookname))
    
    ### Move notebook to report folder
    #time.sleep(5)
    shutil.move(source_to_move, dest)
    print("Report Available at this adress:\n {}".format(dest))
```

```sos kernel="python3" nteract={"transient": {"deleting": false}} outputExpanded=false
create_report(extension = "html", keep_code = True, notebookname = "00_replicate_tables.ipynb")
```

```sos kernel="python3"
### Update TOC in Github
for p in [parent_path,
          str(Path(path).parent),
          #os.path.join(str(Path(path).parent), "00_download_data_from"),
          #os.path.join(str(Path(path).parent.parent), "02_data_analysis"),
          #os.path.join(str(Path(path).parent.parent), "02_data_analysis", "00_statistical_exploration"),
          #os.path.join(str(Path(path).parent.parent), "02_data_analysis", "01_model_estimation"),
         ]:
    try:
        os.remove(os.path.join(p, 'README.md'))
    except:
        pass
    path_parameter = os.path.join(parent_path,'utils', name_json)
    md_lines =  make_toc.create_index(cwd = p, path_parameter = path_parameter)
    md_out_fn = os.path.join(p,'README.md')
    
    if p == parent_path:
    
        make_toc.replace_index(md_out_fn, md_lines, Header = os.path.basename(p).replace('_', ' '), add_description = True, path_parameter = path_parameter)
    else:
        make_toc.replace_index(md_out_fn, md_lines, Header = os.path.basename(p).replace('_', ' '), add_description = False)
```
