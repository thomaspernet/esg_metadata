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
Model estimate Estimate sign of effect


# Description

None

# Metadata

- Key: 242_esg_metadata 
- Epic: Models
- US: Estimate sign of effect
- Task tag: #draft, #polymer, #sign-of-effect
- Analytics reports: 

# Input

## Table/file

**Name**

None

**Github**

- https://github.com/thomaspernet/esg_metadata/blob/master/02_data_analysis/01_model_train_evaluate/01_sign_of_effect/00_sign_of_effect_classification.md


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
import janitor

path = os.getcwd()
parent_path = str(Path(path).parent.parent.parent)


name_credential = 'financial_dep_SO2_accessKeys.csv'
region = 'eu-west-2'
bucket = 'datalake-london'
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
table = 'meta_analysis_esg_cfp'
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
    WITH test as (
  SELECT 
    *, concat(environmental,  social, governance) as filters
  FROM {}.{} 
  WHERE 
    first_date_of_observations IS NOT NULL 
    and last_date_of_observations IS NOT NULL 
    and adjusted_model != 'TO_REMOVE' 
) 
SELECT 
  filters, to_remove, test.id, image, row_id_excel, row_id_google_spreadsheet,
       table_refer, incremental_id, paper_name, publication_name,
       rank, sjr, sjr_best_quartile, h_index, total_docs_2020,
       total_docs_3years, total_refs, total_cites_3years,
       citable_docs_3years, cites_doc_2years, country,
       publication_year, publication_type, cnrs_ranking, peer_reviewed,
       study_focused_on_social_environmental_behaviour, type_of_data,
       first_date_of_observations,mid_year, last_date_of_observations,
       windows, adjusted_model_name,
       adjusted_model, dependent, adjusted_dependent, independent,
       adjusted_independent, 
       social,
       environmental,
       governance,
       financial_crisis,
       kyoto,
       regions,
       study_focusing_on_developing_or_developed_countries,
       lag,
       interaction_term, quadratic_term, n, r2, beta,
       sign_of_effect,
       adjusted_t_value,
       adjusted_standard_error,
       target,
       p_value_significant,
       weight,
       nb_authors,
       reference_count,
       citation_count,
       cited_by_total,
       CASE WHEN is_open_access = TRUE THEN 'YES' ELSE 'NO' END AS is_open_access,
       total_paper,
       esg,
       pct_esg,
       paper_name,
       female,
       male,
       unknown,
       pct_female
FROM 
  test 
  LEFT JOIN (
    SELECT 
      id, 
      COUNT(*) as weight 
    FROM 
      test 
    GROUP BY 
      id
  ) as c on test.id = c.id
  WHERE filters != 'TrueTrueTrue' and filters != 'FalseFalseFalse' and sjr IS NOT NULL

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
df.head(2)
```

```sos kernel="SoS"
df.isna().sum().sort_values().loc[lambda x: x> 0]
```

```sos kernel="SoS"
df['adjusted_model'].unique()
```

```sos kernel="SoS"
df['target'].value_counts()
```

```sos kernel="SoS" nteract={"transient": {"deleting": false}}
pd.DataFrame(schema)
```

<!-- #region kernel="SoS" -->
### Save data to Google Spreadsheet

Data is in [METADATA_MODEL-FINAL_DATA](https://docs.google.com/spreadsheets/d/13gpRy93l7POWGe-rKjytt7KWOcD1oSLACngTEpuqCTg/edit#gid=1219457110)
<!-- #endregion -->

```sos kernel="SoS"
#!pip install --upgrade git+git://github.com/thomaspernet/GoogleDrive-python
```

```sos kernel="python3"
from GoogleDrivePy.google_drive import connect_drive
from GoogleDrivePy.google_authorization import authorization_service
```

```sos kernel="python3"
try:
    os.mkdir("creds")
except:
    pass
```

```sos kernel="SoS"
s3.download_file(key = "CREDS/Financial_dependency_pollution/creds/token.pickle", path_local = "creds")
```

```sos kernel="python3"
import os
auth = authorization_service.get_authorization(
    #path_credential_gcp=os.path.join(parent_path, "creds", "service.json"),
    path_credential_drive=os.path.join(os.getcwd(), "creds"),
    verbose=False,
    scope=['https://www.googleapis.com/auth/spreadsheets.readonly',
           "https://www.googleapis.com/auth/drive"]
)
gd_auth = auth.authorization_drive(path_secret=os.path.join(
    os.getcwd(), "creds", "credentials.json"))
drive = connect_drive.drive_operations(gd_auth)
```

```sos kernel="python3"
import shutil
shutil.rmtree(os.path.join(os.getcwd(),"creds"))
```

```sos kernel="python3"
FILENAME_SPREADSHEET = "METADATA_MODEL"
spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)
```

```sos kernel="python3"
import pandas as pd
from pathlib import Path
path_local = os.path.join(str(Path(os.getcwd()).parent.parent.parent), 
                              "00_data_catalog/temporary_local_data")
output = pd.read_csv( os.path.join(path_local, 'df_meta_analysis_esg_cfp' + '.csv'))
drive.add_data_to_spreadsheet(
    data =output.fillna(""),
    sheetID =spreadsheet_id,
    sheetName = "FINAL_DATA",
    detectRange = True,
    rangeData = None)
```

<!-- #region kernel="SoS" -->
## unbalanced ID
<!-- #endregion -->

```sos kernel="SoS"
df['weight'].describe()
```

```sos kernel="SoS"
(
    df
    .loc[lambda x: x['adjusted_t_value'] <=10 ]
    .reindex(columns = ['adjusted_t_value'])
    .plot
    .hist(10, figsize= (6,6))
)
```

```sos kernel="SoS"
df['adjusted_t_value'].describe()
```

<!-- #region kernel="SoS" -->
## Validation text

"our final database includes 588 studies, divided into 51 journals, 90 titles and 87 different first authors. It is therefore important to note that, among all the studies ultimately selected for our study, 38% of the observations are concentrated in 10 papers and 10 authors"
<!-- #endregion -->

<!-- #region kernel="SoS" -->
- includes 588 studies: CORRECT
<!-- #endregion -->

```sos kernel="SoS"
df.shape[0]
```

<!-- #region kernel="SoS" -->
- divided into 51 journals: It should be 39
<!-- #endregion -->

```sos kernel="SoS"
df['publication_name'].nunique()
```

<!-- #region kernel="SoS" -->
- 90 titles: It should be 78
<!-- #endregion -->

```sos kernel="SoS"
df['id'].nunique()
```

<!-- #region kernel="SoS" -->
- 87 different first authors: TO CHECK
<!-- #endregion -->

<!-- #region kernel="SoS" -->
- 38% of the observations are concentrated in 10 papers: It should be 46
<!-- #endregion -->

```sos kernel="SoS"
(
    (df.groupby('id')['id'].count()/df.shape[0]).rename("count")
    .reset_index()
    .sort_values(by = ['count'], ascending = False)
    .assign(cum_sum = lambda x: x['count'].cumsum())
    .reset_index()
    .drop(columns = ['index'])
    .head(10)
)
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
add_to_dic = False
if add_to_dic:
    if os.path.exists("schema_table.json"):
        os.remove("schema_table.json")
    data = {'to_rename':[], 'to_remove':[]}
    dic_rename = [
        {
        'old':'working\_capital\_i',
        'new':'\\text{working capital}_i'
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

```sos kernel="SoS"
#!conda install -c conda-forge r-lmtest -y
```

```sos kernel="R"
options(warn=-1)
library(tidyverse)
library("sandwich")
library("lmtest")
#library(lfe)
#library(lazyeval)
#library(nnet)
library('progress')
path = "../../../utils/latex/table_golatex.R"
source(path)
```

```sos kernel="R"
%get df_path
df_final <- read_csv(df_path) %>%
mutate_if(is.character, as.factor) %>%

mutate(
    sign_of_effect = relevel(sign_of_effect, ref='NEGATIVE'),
    adjusted_model = relevel(adjusted_model, ref='OTHER'),
    adjusted_dependent = relevel(adjusted_dependent, ref='OTHER'),
      id = as.factor(id),
    governance = relevel(as.factor(governance), ref = 'NO'),
    social = relevel(as.factor(social), ref = 'NO'),
    environmental =relevel(as.factor(environmental), ref = 'NO'),
    financial_crisis =relevel(as.factor(financial_crisis), ref = 'NO'),
    kyoto =relevel(as.factor(kyoto), ref = 'NO'),
    target =relevel(as.factor(target), ref = 'NOT_SIGNIFICANT'),
    study_focusing_on_developing_or_developed_countries =relevel(
        as.factor(study_focusing_on_developing_or_developed_countries), ref = 'WORLDWIDE'),
    regions =relevel(as.factor(regions), ref = 'WORLDWIDE'),
    cnrs_ranking =relevel(as.factor(cnrs_ranking), ref = '0'),
    is_open_access =relevel(as.factor(is_open_access), ref = 'NO'),
)
```

```sos kernel="R"
glimpse(df_final)
```

```sos kernel="R"
transpose(df_final %>% 
    select_if(function(x) any(is.na(x))) %>% 
    summarise_each(funs(sum(is.na(.)))))
```

<!-- #region kernel="R" -->
GLM does not clustered the standard error so, we compute it by hand
<!-- #endregion -->

```sos kernel="R"
se_robust <- function(x)
  coeftest(x, vcov. = sandwich::sandwich
          )[, 2]
```

<!-- #region kernel="R" -->
# Table 1: Probit

$$
\mathrm{P}\left(\text { Significant }_{\mathrm{ib}}=\mathrm{significant}\right)=\mathrm{\beta}_{0} + 
\mathrm{\beta}_{1}\text { ESG }_{\mathrm{ib}}+ 
\mathrm{\beta}_{2}\text { Kyoto }_{\mathrm{i}} +
\mathrm{\beta}_{3}\text { Financial crisis }_{\mathrm{i}} +
\mathrm{\beta}_{4}\text { Publication year }_{\mathrm{i}} + 
\mathrm{\beta}_{5}\text { windows }_{\mathrm{i}} +
\mathrm{\beta}_{6}\text { mid-year }_{\mathrm{i}} +
\mathrm{\beta}_{7}\text { region }_{\mathrm{ib}}
+\epsilon _{\mathrm{ib}}
$$

- robust standard error
- Cannot compute clustered standard error if we add features without variation among the c luster (i.e `n`, or journal information)

## Variable construction


* Significant: If in the table, p-value below .1, then significant else not significant
* The variable adjusted_independent is too imbalanced, and we are interested in only:
  * SOCIAL
  * ENVIRONMENTAL
  * GOVERNANCE
* So need to create three underlying dummy variables: rules below
  * Source low-level variable: https://docs.google.com/spreadsheets/d/1d66_CVtWni7wmKlIMcpaoanvT2ghmjbXARiHgnLWvUw/edit#gid=146632716&range=B126
  * SOCIAL if adjusted_independent : 
    * ENVIRONMENTAL AND SOCIAL
    * SOCIAL
    * CSP
    * CSR
    * ENVIRONMENTAL, SOCIAL and GOVERNANCE
  * ENVIRONMENTAL if adjusted_independent :
    * ENVIRONMENTAL
    * ENVIRONMENTAL AND SOCIAL
    * ENVIRONMENTAL, SOCIAL and GOVERNANCE
  * GOVERNANCE if adjusted_independent :
    * GOVERNANCE
    * ENVIRONMENTAL, SOCIAL and GOVERNANCE
- adjusted_model: https://docs.google.com/spreadsheets/d/1d66_CVtWni7wmKlIMcpaoanvT2ghmjbXARiHgnLWvUw/edit#gid=793443705&range=B34
- adjusted_dependent: https://docs.google.com/spreadsheets/d/1d66_CVtWni7wmKlIMcpaoanvT2ghmjbXARiHgnLWvUw/edit#gid=450174628&range=B59
- Region:
    - AFRICA: 'Cameroon', 'Egypt', 'Libya', 'Morocco', 'Nigeria'
    - ASIA AND PACIFIC:  'India', 'Indonesia', 'Taiwan', 'Vietnam', 
        'Australia', 'China', 'Iran', 'Malaysia', 
        'Pakistan', 'South Korea', 'Bangladesh'
    - EUROPE: 'Spain', '20 European countries', 
        'United Kingdom', 'France', 'Germany, Italy, the Netherlands and United Kingdom', 
        'Turkey', 'UK'
    - LATIN AMERICA: 'Latin America', 'Brazil'
    - NORTH AMERICA: 'USA', 'US', 'U.S.', 'Canada'
    - ELSE WORLDWIDE
- Kyoto first_date_of_observations >= 1997 THEN TRUE ELSE FALSE ,
- Financial crisis first_date_of_observations >= 2009 THEN TRUE ELSE FALSE 
- windows: last_date_of_observations - first_date_of_observations
- mid-year: last_date_of_observations - (windows/2)


## note about Probit 

TO estimate a probit, use `probit` link function.  For logistic regression, use `binomial`

- Reason Probit instead of Logit
    - [What is the Difference Between Logit and Probit Models?](https://tutorials.methodsconsultants.com/posts/what-is-the-difference-between-logit-and-probit-models/)

Logit and probit differ in how they define $f(∗)$. The logit model uses something called the cumulative distribution function of the logistic distribution. The probit model uses something called the cumulative distribution function of the standard normal distribution to define $f(∗)$.

Probit models can be generalized to account for non-constant error variances in more advanced econometric settings (known as heteroskedastic probit models)

## How to read

**Comparison group**

- Always `OTHER`
- Target: `SIGNIFICANT`
- regions: `WORLDWIDE`
- cnrs_ranking: `0`

**Odd ratio**

- Categorical:
    - Keeping all other variables constant, if the analysis uses FIXED EFFECT model, there are 2.71 times more likely to stay in the NEGATIVE sign category as compared to the OTHER model category. The coefficient, however, is not significant. (Col 1)
- Continuous:
    - Keeping all other variables constant, if the SJR score increases one unit, there is 1.003 times more likely to stay in the POSITIVE sign category as compared to the OTHER model category y (the risk or odds is .2% higher). The coefficient is significant.
    
Here, OTHER means insignificant
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))
### Econometrics control
t_3 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje")
```

```sos kernel="R"
### Baseline SJR
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))
### Econometrics control
t_3 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje")
```

<!-- #region kernel="R" -->
### Papers and authors specification

<!-- #endregion -->

```sos kernel="R"
normalit<-function(m){
   (m - min(m))/(max(m)-min(m))
 }

df_final <- df_final %>% mutate(
    pct_esg_1 = normalit(pct_esg),
    esg_1 =  normalit(esg)
)
```

```sos kernel="R"
###
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))


t_1 <- glm(target ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_1.rrr <- exp(coef(t_1))

t_2 <- glm(target ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_2.rrr <- exp(coef(t_2))

list_final = list(t_0, t_1, t_2)
list_final.rrr = list(t_0.rrr,t_1.rrr,t_2.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="table1.txt")
```

```sos kernel="R"
###
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))


t_1 <- glm(target ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_1.rrr <- exp(coef(t_1))

t_2 <- glm(target ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_2.rrr <- exp(coef(t_2))

list_final = list(t_0, t_1, t_2)
list_final.rrr = list(t_0.rrr,t_1.rrr,t_2.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="table2.txt")
```

<!-- #region kernel="SoS" -->
## Model OLS: 

$$
 \text{T-value}=\mathrm{\beta}_{0} + 
\mathrm{\beta}_{1}\text { ESG }_{\mathrm{ib}}+ 
\mathrm{\beta}_{2}\text { Kyoto }_{\mathrm{i}} +
\mathrm{\beta}_{3}\text { Financial crisis }_{\mathrm{i}} +
\mathrm{\beta}_{4}\text { Publication year }_{\mathrm{i}} + 
\mathrm{\beta}_{5}\text { windows }_{\mathrm{i}} +
\mathrm{\beta}_{6}\text { mid-year }_{\mathrm{i}} +
\mathrm{\beta}_{7}\text { region }_{\mathrm{ib}}
+\epsilon _{\mathrm{ib}}
$$

### Computation t-value

* construct should_t_value   equals to “TO_CHECK” → if test_standard_error   = “TO_CHECK” and adjusted_model  is not PANEL or POOLED (use panel because panel use clustered/robust standard error no direct computation), then check if switch standard error and t-stat, so use column sr has t-stat and compare with critical value. If match critical value, and equals to stars  then OK, else “TO_CHECK”
* Construct adjusted_standard_error : if test_standard_error  is OK and should_t_value  is NO_NEED_TO_CHECK then use sr , else leave blank
* Construct **adjusted_t_value**: 
  * ⚠️ critical value (the raw data has a column for the t_value which is similar, but the variable adjusted_t_value is reconstructed based on known t_value or in case of unknown t_value then from standard error or p-value: 
    * If test_t_value is equals to TO_CHECK or OK then use t_value ← We use the value reported in the paper, not the one reconstructed
    * ELSE if test_standard_error is equal to NO_SE and test_p_value is equal to OK then we can compute the critical value using the t-inverse function. 
      * Ex: round(T.INV(1-X114, I114) where X114 is the p-value, so we want to get the right tail. If p-value is .05, the the right tail is .95.
    * ELSE beta / standard error 
    * Note, if critical value cannot be computed, it is because of one of the following reason
      *  p-value is 0, then cannot compute the critical value
      * standard error is 0, cannot divide by 0
      * missing standard error, t-value or p-value
      
- Remove 10 outliers -> critical value more than 1K -> high leverage and does not represent the true data
- Standard error robust
<!-- #endregion -->

<!-- #region kernel="SoS" -->
Model 1: No absolute value

Interested in the magnitude of the t-student critical value
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

```sos kernel="R"
### Baseline SJR
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

<!-- #region kernel="R" -->
Author
<!-- #endregion -->

```sos kernel="R"
###
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )
t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term
           + is_open_access
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

list_final = list(t_0, t_1, t_2)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

<!-- #region kernel="R" -->
Model 2: absolute value

Interested in the factors leading to larger t-student critical value, hence significant coefficient
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(abs(adjusted_t_value) ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(abs(adjusted_t_value) ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(abs(adjusted_t_value) ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_4 <- glm(abs(adjusted_t_value) ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

```sos kernel="R"
### Baseline SJR
t_0 <- glm(abs(adjusted_t_value) ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(abs(adjusted_t_value) ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(abs(adjusted_t_value) ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_4 <- glm(abs(adjusted_t_value) ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + cnrs_ranking
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

<!-- #region kernel="R" heading_collapsed="true" -->
# Statistics

## Target

<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby("environmental")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "environment"}),
                            (
                                df.groupby("social")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "social"}),
                            (
                                df.groupby("governance")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "governance"}),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"environmental": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby(["environmental", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "environment"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["social", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "social"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["governance", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "governance"})
                                .unstack(0)
                            ),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"environmental": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("adjusted_model")
                    .agg({"target": "value_counts"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("target", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["adjusted_model", "target"])
                    .agg({"id": "nunique"})
                    .rename(columns={"id": "adjusted_model"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("adjusted_model", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

```sos kernel="SoS"
(
    pd.concat(
        [
            (df.groupby("kyoto").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'kyoto'}),
            (df.groupby("financial_crisis").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'financial_crisis'}),
        ],
        axis=1,
    )
    .T
    .reset_index()
    .rename(columns = {'kyoto':'is_dummy', 'level_0':'origin'})
    .set_index(['origin','is_dummy'])
    .assign(pct_significant = lambda x: x[('SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby("kyoto")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "kyoto"}),
                            (
                                df.groupby("financial_crisis")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "financial_crisis"}),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"kyoto": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby(["kyoto", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "kyoto"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["financial_crisis", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "financial_crisis"})
                                .unstack(0)
                            ),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"kyoto": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

<!-- #region kernel="R" -->
## Papers
<!-- #endregion -->

```sos kernel="SoS"
(
    df
    .groupby('target')
    .agg(
    {
        'windows':'describe'
    })
)
```

<!-- #region kernel="R" -->
- lag
- interaction_term
- quadratic_term
<!-- #endregion -->

```sos kernel="SoS"
(
    pd.concat(
        [
            (df.groupby("lag").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'lag'}),
            (df.groupby("interaction_term").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'interaction_term'}),
            (df.groupby("quadratic_term").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'quadratic_term'}),
        ],
        axis=1,
    )
    .T
    .reset_index()
    .rename(columns = {'lag':'is_dummy', 'level_0':'origin'})
    .set_index(['origin','is_dummy'])
    .assign(pct_significant = lambda x: x[('SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby("lag")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "lag"}),
                            (
                                df.groupby("interaction_term")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "interaction_term"}),
                            (
                                df.groupby("quadratic_term")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "quadratic_term"}),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"lag": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    pd.concat(
                        [
                            (
                                df.groupby(["lag", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "lag"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["interaction_term", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "interaction_term"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["quadratic_term", "target"])
                                .agg({"id": "nunique"})
                                .rename(columns={"id": "quadratic_term"})
                                .unstack(0)
                            ),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"lag": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

<!-- #region kernel="R" -->
##  Region

- regions
- study_focusing_on_developing_or_developed_countries

Comparison group: "WORLDWIDE"
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("regions")
                    .agg({"target": "value_counts"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("target", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["regions", "target"])
                    .agg({"id": "nunique"})
                    .rename(columns={"id": "regions"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("regions", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

<!-- #region kernel="R" -->
## Journal

- sjr 
- sjr_best_quartile: Q1
- cnrs_ranking: 0
- h_index
<!-- #endregion -->

```sos kernel="SoS"
(
    df
    .groupby('target')
    .agg(
    {
        'sjr':'describe'
    })
    #.unstack(-1)
    #.assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("cnrs_ranking")
                    .agg({"target": "value_counts"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("target", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["count"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["cnrs_ranking", "target"])
                    .agg({"id": "nunique"})
                    .rename(columns={"id": "cnrs_ranking"})
                    .unstack(-1)
                    .assign(
                        pct_significant=lambda x: x[("cnrs_ranking", "SIGNIFICANT")]
                        / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
)
```

```sos kernel="SoS"
(
    df
    .groupby('target')
    .agg(
    {
        'h_index':'describe'
    })
    #.unstack(-1)
    #.assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
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
import create_report
```

```sos kernel="python3"
name_json = 'parameters_ETL_esg_metadata.json'
path_json = os.path.join(str(Path(path).parent.parent), 'utils',name_json)
```

```sos kernel="python3" nteract={"transient": {"deleting": false}} outputExpanded=false
create_report.create_report(extension = "html", keep_code = True, notebookname = "00_sign_of_effect_classification.ipynb")
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
