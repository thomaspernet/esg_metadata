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
       first_date_of_observations, last_date_of_observations,
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
       weight
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
## unbalanced ID
<!-- #endregion -->

```sos kernel="SoS"
df['weight'].describe()
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
se_robust_clustered <- function(x)
  coeftest(x,
         vcov. = vcovCL(t_2, cluster = df_final %>% filter(adjusted_model != 'TO_REMOVE') %>% select(id), type = "HC0")
        )[, 2]
```

<!-- #region kernel="R" -->
# Table 1: Baseline

$$
\begin{aligned}
\text{Write your equation}
\end{aligned}
$$

- robust standard error
- Cannot compute clustered standard error if we add features without variation among the cluster (i.e `n`, or journal information)

TO estimate a probit, use `probit` link function.  For logistic regression, use `binomial`

- Reason Probit instead of Logit
    - [What is the Difference Between Logit and Probit Models?](https://tutorials.methodsconsultants.com/posts/what-is-the-difference-between-logit-and-probit-models/)

Logit and probit differ in how they define $f(∗)$. The logit model uses something called the cumulative distribution function of the logistic distribution. The probit model uses something called the cumulative distribution function of the standard normal distribution to define $f(∗)$.

Probit models can be generalized to account for non-constant error variances in more advanced econometric settings (known as heteroskedastic probit models)

**Comparison group**

- Always `OTHER`
- Target: `SIGNIFICANT`

**How to read**

- Categorical:
    - Keeping all other variables constant, if the analysis uses FIXED EFFECT model, there are 2.71 times more likely to stay in the NEGATIVE sign category as compared to the OTHER model category. The coefficient, however, is not significant. (Col 1)
- Continuous:
    - Keeping all other variables constant, if the SJR score increases one unit, there is 1.003 times more likely to stay in the POSITIVE sign category as compared to the OTHER model category y (the risk or odds is .2% higher). The coefficient is significant.
    
Here, OTHER means insignificant

Currently, issue with:

- governance 
- full inclusion dummy -> probably collinearity need to check

Test with Kyoto, financial crisis & region

- CASE WHEN first_date_of_observations >= 1997 THEN TRUE ELSE FALSE END AS kyoto,
- CASE WHEN first_date_of_observations >= 2009 THEN TRUE ELSE FALSE END AS financial_crisis
<!-- #endregion -->

```sos kernel="SoS"
(
    pd.concat(
        [
            (df.groupby("environmental").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'environment'}),
            (df.groupby("social").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'social'}),
            (df.groupby("governance").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'governance'}),
        ],
        axis=1,
    )
    .T
    .reset_index()
    .rename(columns = {'environmental':'is_dummy', 'level_0':'origin'})
    .set_index(['origin','is_dummy'])
    .assign(pct_significant = lambda x: x[('SIGNIFICANT')]/x.sum(axis= 1))
    #
    #
)
```

```sos kernel="SoS"
(
    df
    .groupby('adjusted_model')
    .agg(
    {
        'target':'value_counts'
    })
    .unstack(-1)
    .assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
(
    pd.concat(
        [
            (df.groupby("kyoto").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'kyoto'}),
            (df.groupby("financial_crisis").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'financial_crisis'}),
            #(df.groupby("governance").agg({"target": "value_counts"}).unstack(0)).rename(columns = {'target':'governance'}),
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

```sos kernel="R"
t_0 <- glm(target ~ environmental,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~social,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance ,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

### add model
t_3 <- glm(target ~ environmental+
           adjusted_model  ,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social+
           adjusted_model,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           +adjusted_model,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

### add kyoto and financial crisis
t_6 <- glm(target ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis,
           data = df_final ,
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis,
           data = df_final , binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           +adjusted_model
                + kyoto 
                + financial_crisis,
           data = df_final , binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr,t_6.rrr,t_7.rrr,t_8.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          omit = "id", style = "qje")
```

<!-- #region kernel="SoS" -->
## Model OLS: 

- Remove 10 outliers -> critical value more than 1K -> high leverage and does not represent the true data
- Standard error robust

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
<!-- #endregion -->

```sos kernel="SoS"
df['adjusted_t_value'].describe()
```

<!-- #region kernel="SoS" -->
Model 1: No absolute value

Interested in the magnitude of the t-student critical value
<!-- #endregion -->

```sos kernel="R"
t_0 <- glm(adjusted_t_value ~ environmental,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))
t_1 <- glm(adjusted_t_value ~social,
           data = df_final %>% filter(adjusted_t_value < 10) , family=gaussian(identity))
t_2 <- glm(adjusted_t_value ~ governance ,
           data = df_final %>% filter(adjusted_t_value < 10) , family=gaussian(identity))

### add model
t_3 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
          
t_4 <- glm(adjusted_t_value ~ social+
           adjusted_model,
           data = df_final %>% filter(adjusted_t_value < 10) , family=gaussian(identity))
t_5 <- glm(adjusted_t_value ~ governance
           +adjusted_model,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))

### add kyoto and financial crisis
t_6 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
          
t_7 <- glm(adjusted_t_value ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis,
           data = df_final  %>%filter(adjusted_t_value < 10) ,family=gaussian(identity))
t_8 <- glm(adjusted_t_value ~ governance
           +adjusted_model
                + kyoto 
                + financial_crisis,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
Model 2: absolute value

Interested in the factors leading to larger t-student critical value, hence significant coefficient
<!-- #endregion -->

```sos kernel="R"
t_0 <- glm(abs(adjusted_t_value) ~ environmental,
           data = df_final %>%filter(adjusted_t_value < 10),
           family=gaussian(identity))
t_1 <- glm(abs(adjusted_t_value) ~social,
           data = df_final %>%filter(adjusted_t_value < 10), family=gaussian(identity))
t_2 <- glm(abs(adjusted_t_value) ~ governance ,
           data = df_final %>%filter(adjusted_t_value < 10), family=gaussian(identity))

### add model
t_3 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  ,
           data = df_final %>%filter(adjusted_t_value < 10),
           family=gaussian(identity))
          
t_4 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model,
           data = df_final %>%filter(adjusted_t_value < 10), family=gaussian(identity))
t_5 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model,
           data = df_final %>%filter(adjusted_t_value < 10), family=gaussian(identity))

### add kyoto and financial crisis
t_6 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis,
           data = df_final %>%filter(adjusted_t_value < 10),
           family=gaussian(identity))
          
t_7 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis,
           data = df_final %>%filter(adjusted_t_value < 10),family=gaussian(identity))
t_8 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
                + kyoto 
                + financial_crisis,
           data = df_final %>%filter(adjusted_t_value < 10), family=gaussian(identity))


list_final = list(t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
# Table 2: Paper control

- Add the following control:
    - publication_year
    - first_date_of_observations
    - last_date_of_observations
    - windows
    - avg_windows
    - lag
    - interaction_term
    - quadratic_term
<!-- #endregion -->

```sos kernel="SoS"
(
    df
    .groupby('target')
    .agg(
    {
        'windows':'describe'
    })
    #.unstack(-1)
    #.assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="R"
t_0 <- glm(target ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations
           ,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

### window 
t_3 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr ,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
## OLS
<!-- #endregion -->

```sos kernel="R"
t_0 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
t_1 <- glm(adjusted_t_value ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))


### window 
t_3 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
t_4 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)

stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

```sos kernel="R"
t_0 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
t_1 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + publication_year
           + first_date_of_observations
           + last_date_of_observations,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))


### window 
t_3 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))
t_4 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year
           + windows,
           data = df_final  %>%filter(adjusted_t_value < 10) , family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)

stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
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

```sos kernel="R"
### lag
t_0 <- glm(target ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + lag
           ,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + lag,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + lag,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

### interaction_term 
t_3 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

### quadratic_term
t_6 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final ,
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))

list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr ,t_5.rrr,t_6.rrr,t_7.rrr ,t_8.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
## OLS
<!-- #endregion -->

```sos kernel="R"
### lag
t_0 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + lag
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + lag,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + lag,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### interaction_term 
t_3 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### quadratic_term
t_6 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_7 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_8 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8)

stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

```sos kernel="R"
### lag
t_0 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + lag
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + lag,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + lag,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### interaction_term 
t_3 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + interaction_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### quadratic_term
t_6 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_7 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_8 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + quadratic_term,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8)

stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
# Table 3: Region

- regions
- study_focusing_on_developing_or_developed_countries

Comparison group: "WORLDWIDE"
<!-- #endregion -->

```sos kernel="SoS"
(
    df
    .groupby('regions')
    .agg(
    {
        'target':'value_counts'
    })
    .unstack(-1)
    .assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
(
    df
    .groupby('study_focusing_on_developing_or_developed_countries')
    .agg(
    {
        'target':'value_counts'
    })
    .unstack(-1)
    .assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="R"
### regions
t_0 <- glm(target ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + regions
           ,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + regions,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + regions,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

### study_focusing_on_developing_or_developed_countries 
t_3 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr ,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
## OLS
<!-- #endregion -->

```sos kernel="R"
### regions
t_0 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + regions
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + regions,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + regions,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### study_focusing_on_developing_or_developed_countries 
t_3 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))



list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

```sos kernel="R"
### regions
t_0 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + regions
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + regions,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + regions,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### study_focusing_on_developing_or_developed_countries 
t_3 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + study_focusing_on_developing_or_developed_countries,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))



list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
# Table 4: Journal

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
(
    df
    .groupby('sjr_best_quartile')
    .agg(
    {
        'target':'value_counts'
    })
    .unstack(-1)
    .assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
)
```

```sos kernel="SoS"
(
    df
    .groupby('cnrs_ranking')
    .agg(
    {
        'target':'value_counts'
    })
    .unstack(-1)
    .assign(pct_significant = lambda x: x[('target','SIGNIFICANT')]/x.sum(axis= 1))
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

```sos kernel="R"
### sjr
t_0 <- glm(target ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + sjr
           ,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + sjr,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + sjr,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

### sjr_best_quartile 
t_3 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + sjr_best_quartile ,
           data = df_final %>% filter(sjr_best_quartile != 'Q3'),
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final %>% filter(sjr_best_quartile != 'Q3'), binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final %>% filter(sjr_best_quartile != 'Q3'), binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

### cnrs_ranking
t_6 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final ,
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final , binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final , binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))

### h_index
t_9 <- glm(target ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final ,
           binomial(link = "probit")
          )
t_9.rrr <- exp(coef(t_9))
t_10 <- glm(target ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final , binomial(link = "probit"))
t_10.rrr <- exp(coef(t_10))
t_11 <- glm(target ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final , binomial(link = "probit"))
t_11.rrr <- exp(coef(t_11))

list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8, t_9,t_10, t_11)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr ,t_5.rrr,t_6.rrr,t_7.rrr ,t_8.rrr,t_9.rrr,t_10.rrr ,t_11.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          omit = "id", style = "qje")
```

<!-- #region kernel="R" -->
## OLS
<!-- #endregion -->

```sos kernel="R"
### sjr
t_0 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + sjr
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + sjr,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + sjr,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### sjr_best_quartile 
t_3 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + sjr_best_quartile ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### cnrs_ranking
t_6 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_7 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_8 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### h_index
t_9 <- glm(adjusted_t_value ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_10 <- glm(adjusted_t_value ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_11 <- glm(adjusted_t_value ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8, t_9,t_10, t_11)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
```

```sos kernel="R"
### sjr
t_0 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
                + kyoto 
                + financial_crisis
           + sjr
           ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_1 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
                + kyoto 
                + financial_crisis
           + sjr,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_2 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
            + kyoto 
            + financial_crisis
           + sjr,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### sjr_best_quartile 
t_3 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + sjr_best_quartile ,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_4 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_5 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + sjr_best_quartile,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### cnrs_ranking
t_6 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_7 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_8 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + cnrs_ranking,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


### h_index
t_9 <- glm(abs(adjusted_t_value) ~ environmental+
           adjusted_model  
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_10 <- glm(abs(adjusted_t_value) ~ social+
           adjusted_model    
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))

t_11 <- glm(abs(adjusted_t_value) ~ governance
           +adjusted_model
           + kyoto 
           + financial_crisis
           + h_index,
           data = df_final  %>%filter(adjusted_t_value < 10) ,
           family=gaussian(identity))


list_final = list(t_0, t_1, t_2,t_3, t_4, t_5, t_6,t_7, t_8, t_9,t_10, t_11)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          omit = "id", style = "qje")
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
