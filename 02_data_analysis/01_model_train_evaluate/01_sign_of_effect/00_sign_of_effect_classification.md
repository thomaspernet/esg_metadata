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
import seaborn as sns
import matplotlib.pyplot as plt
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
filename = "df_{}".format(table)
full_path_filename = "SQL_OUTPUT_ATHENA/CSV/{}.csv".format(filename)
path_local = os.path.join(
    str(Path(path).parent.parent.parent), "00_data_catalog/temporary_local_data"
)
df_path = os.path.join(path_local, filename + ".csv")
if download_data:

    s3 = service_s3.connect_S3(client=client, bucket=bucket, verbose=False)
    query = """
    WITH test as (
  SELECT 
    *, concat(environmental,  social, governance) as filters
  FROM esg.meta_analysis_esg_cfp
  WHERE 
    first_date_of_observations IS NOT NULL 
    and last_date_of_observations IS NOT NULL 
    and adjusted_model != 'TO_REMOVE' 
) 
SELECT 
  filters,
 paperid,
 nb_authors,
 reference_count,
 citation_count,
 influential_citation_count,
  CASE WHEN is_open_access = TRUE THEN 'YES' ELSE 'NO' END AS is_open_access,
 total_paper,
 esg,
 pct_esg,
 test.id_source,
 female,
 male,
 unknown,
 pct_female,
 drive_url,
 image,
 row_id_google_spreadsheet,
 table_refer,
 adjusted_model,
 adjusted_dependent,
 adjusted_independent,
 social,
 environmental,
 governance,
 lag,
 interaction_term,
 quadratic_term,
 n,
 target,
 adjusted_standard_error,
 adjusted_t_value,
 paper_name,
 first_date_of_observations,
 last_date_of_observations,
 csr_20_categories,
 kyoto,
 financial_crisis,
 windows,
 mid_year,
 regions,
 providers,
 publication_year,
 publication_name,
 rank_digit,
 CASE WHEN cluster_w_emb = 0 THEN 'CLUSTER_0'
       WHEN cluster_w_emb = 1 THEN 'CLUSTER_1'
       ELSE 'CLUSTER_2' END AS cluster_w_emb,
 sentiment,
 lenght,
 adj,
 noun,
 verb,
 size_abstract,
 pct_adj,
 pct_noun,
 pct_verb,
 rank,
 sjr,
 region_journal,
 weight
FROM 
  test 
  LEFT JOIN (
    SELECT 
      id_source, 
      COUNT(*) as weight 
    FROM 
      test 
    GROUP BY 
      id_source
  ) as c on test.id_source = c.id_source
  WHERE filters != 'TrueTrueTrue' and filters != 'FalseFalseFalse' and regions != 'ARAB WORLD'
    """.format(
        db, table
    )
    try:
        df = (s3.run_query(
            query=query,
            database=db,
            s3_output="SQL_OUTPUT_ATHENA",
            filename=filename,  # Add filename to print dataframe
            destination_key="SQL_OUTPUT_ATHENA/CSV",  # Use it temporarily
            dtype=dtypes,
        ).assign(
            d_rank_digit=lambda x: np.where(
                x["rank_digit"].isin(["1"]), "rank_1", "rank_2345"
            ),
            publication_year_int=lambda x: pd.factorize(x["publication_year"])[0],
            interaction_term = lambda x: x['interaction_term'].str.strip()
        ))
    except:
        pass
(df.to_csv(os.path.join(path_local, "df_meta_analysis_esg_cfp" + ".csv")))
df = pd.read_csv(os.path.join(path_local, "df_meta_analysis_esg_cfp" + ".csv"))
```

```sos kernel="SoS"
df.shape
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

```sos kernel="SoS"
try:
    os.mkdir(os.path.join(os.getcwd(),"creds"))
except:
    pass

s3.download_file(key = "CREDS/Financial_dependency_pollution/creds/token.pickle",
                     path_local = "creds")
```

```sos kernel="python3"
from GoogleDrivePy.google_drive import connect_drive
from GoogleDrivePy.google_authorization import authorization_service
import os
import shutil
import pandas as pd
from pathlib import Path

auth = authorization_service.get_authorization(
        path_credential_gcp=os.path.join(os.getcwd(), "creds", "service.json"),
        path_credential_drive=os.path.join(os.getcwd(), "creds"),
        verbose=False,
        scope=['https://www.googleapis.com/auth/spreadsheets.readonly',
               "https://www.googleapis.com/auth/drive"]
    )
gd_auth = auth.authorization_drive(path_secret=os.path.join(
        os.getcwd(), "creds", "credentials.json"))
drive = connect_drive.drive_operations(gd_auth)
shutil.rmtree(os.path.join(os.getcwd(),"creds"))

move_g_spreadsheet = False
if move_g_spreadsheet:
    FILENAME_SPREADSHEET = "METADATA_MODEL"
    spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)

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

<!-- #region kernel="python3" -->
# Statisitcs
<!-- #endregion -->

<!-- #region kernel="python3" -->
## Basic information
<!-- #endregion -->

<!-- #region kernel="python3" -->
- Number of observations
<!-- #endregion -->

```sos kernel="SoS"
df.shape[0]
```

<!-- #region kernel="SoS" -->
- Number of Journals
<!-- #endregion -->

```sos kernel="SoS"
df['publication_name'].nunique()
```

<!-- #region kernel="SoS" -->
- Number of publications
<!-- #endregion -->

```sos kernel="SoS"
df['paperid'].nunique()
```

<!-- #region kernel="SoS" -->
- Number of Authors
<!-- #endregion -->

```sos kernel="python3"
FILENAME_SPREADSHEET = "AUTHOR_SEMANTIC_GOOGLE"
spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)
df_author = (
    drive.download_data_from_spreadsheet(
    sheetID = spreadsheet_id,
    sheetName = "Sheet1",
    to_dataframe = True)
    .to_csv('temp_author.csv', index= False)
)
```

```sos kernel="SoS"
(
    pd.read_csv('temp_author.csv')
    .loc[lambda x: x['paperId'].isin(df['paperid'].unique())]
    .reindex(columns = ['name'])
    .drop_duplicates()
    .count()
)
```

<!-- #region kernel="SoS" -->
- Number of papers per author
<!-- #endregion -->

```sos kernel="SoS"
(
    pd.read_csv('temp_author.csv')
    .loc[lambda x: x['paperId'].isin(df['paperid'].unique())]
    .groupby('name')
    .agg(
    {
        'paperId':'nunique'
    })
    .sort_values(by = ['paperId'])
    .groupby('paperId')
    .agg(
        {
            'paperId':'count'
        }
    )
    .rename(columns = {'paperId':'count'})
    
)
```

<!-- #region kernel="SoS" -->
- unbalanced ID
<!-- #endregion -->

```sos kernel="SoS"
(
    df
    .reindex(columns = ['weight'])
    .plot
    .hist(5, figsize= (6,6))
)
```

```sos kernel="SoS"
(
    (df.groupby('id_source')['id_source'].count()/df.shape[0]).rename("count")
    .reset_index()
    .sort_values(by = ['count'], ascending = False)
    .assign(cum_sum = lambda x: x['count'].cumsum())
    .reset_index()
    .drop(columns = ['index', 'count', 'id_source'])
    .plot
    .line(title = "cumulated number of observations per paper",figsize= (6,6))
    
    #.head(10)
)
```

<!-- #region kernel="python3" -->
## Statistic baseline

- environmental 
- social 
- governance
- adjusted_model  
- kyoto 
- financial_crisis
- publication_year
- windows
- mid_year
- regions
- sjr
- is_open_access
- region_journal
- providers
<!-- #endregion -->

```sos kernel="SoS"
for v in [
    "target",
    "environmental",
    "social",
    "governance",
    "adjusted_model",
    "kyoto",
    "financial_crisis",
    "publication_year",
    "regions",
    "is_open_access",
    "region_journal",
    "providers",
]:
    print("\n\nDisplay variable: {}\n\n".format(v))
    display(
        pd.concat([df[v].value_counts(), df[v].value_counts(normalize=True)], axis=1)
    )
```

```sos kernel="SoS"
for v in ['publication_year', "windows", "mid_year", "sjr"]:
    #print("\n\nDisplay variable: {}\n\n".format(v))
    (
        df
        .reindex(columns = [v])
        .plot
        .hist(10, figsize= (6,6), title = "{} From {} to {}".format(
            v,
            df[v].min(),
            df[v].max()
        ))
    )
```

<!-- #region kernel="SoS" -->
### Distribution baseline feature with target

"adjusted_model",  
"kyoto" ,
"financial_crisis",
"publication_year",
"windows",
"mid_year",
"regions",
"sjr",
"is_open_access",
"region_journal",
"providers"
<!-- #endregion -->

<!-- #region kernel="SoS" -->
#### environmental, social, governance
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
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
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
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "environment"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["social", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "social"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["governance", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "governance"})
                                .unstack(0)
                            ),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"environmental": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
).style.format("{:,.2%}", subset = [
    ('count','pct_significant'),
    ('count','pct_total'),
    ('paper count','pct_significant'),
    ('paper count','pct_total'),
])
```

<!-- #region kernel="SoS" -->
#### adjusted_model
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("adjusted_model")
                    .agg({"adjusted_model": "count"})
                    .rename(columns={"adjusted_model": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("adjusted_model")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "adjusted_model"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("adjusted_model")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                
                (
                    df.groupby(["adjusted_model", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "adjusted_model"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T.assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1)
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### kyoto, financial_crisis
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
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
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
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "kyoto"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["financial_crisis", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "financial_crisis"})
                                .unstack(0)
                            )
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"kyoto": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
).style.format("{:,.2%}", subset = [
    ('count','pct_significant'),
    ('count','pct_total'),
    ('paper count','pct_significant'),
    ('paper count','pct_total')
])
```

<!-- #region kernel="SoS" -->
#### publication_year
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("publication_year")
                    .agg({"publication_year": "count"})
                    .rename(columns={"publication_year": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("publication_year")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "publication_year"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("publication_year")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["publication_year", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "publication_year"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### is_open_access
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
                                df.groupby("is_open_access")
                                .agg({"target": "value_counts"})
                                .unstack(0)
                            ).rename(columns={"target": "is_open_access"}),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"is_open_access": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
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
                                df.groupby(["is_open_access", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "is_open_access"})
                                .unstack(0)
                            )
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"is_open_access": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
).style.format("{:,.2%}", subset = [
    ('count','pct_significant'),
    ('count','pct_total'),
    ('paper count','pct_significant'),
    ('paper count','pct_total')
])
```

<!-- #region kernel="SoS" -->
#### region_journal
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("region_journal")
                    .agg({"region_journal": "count"})
                    .rename(columns={"region_journal": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("region_journal")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "region_journal"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("region_journal")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["region_journal", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "region_journal"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### providers
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("providers")
                    .agg({"providers": "count"})
                    .rename(columns={"providers": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("providers")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "providers"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("providers")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["providers", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "providers"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### regions
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("regions")
                    .agg({"regions": "count"})
                    .rename(columns={"regions": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("regions")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "regions"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("regions")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["regions", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "regions"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### sjr
<!-- #endregion -->

```sos kernel="SoS"
(df.groupby("target").agg({"sjr": "describe"}))
```

<!-- #region kernel="SoS" -->
#### CNRS
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("rank_digit")
                    .agg({"rank_digit": "count"})
                    .rename(columns={"rank_digit": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("rank_digit")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "rank_digit"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("rank_digit")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["rank_digit", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "rank_digit"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

```sos kernel="SoS"
sorted(list(df.loc[lambda x: x['rank_digit'].isin(['5'])]['publication_name'].unique()))
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("d_rank_digit")
                    .agg({"d_rank_digit": "count"})
                    .rename(columns={"d_rank_digit": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("d_rank_digit")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "d_rank_digit"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("d_rank_digit")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["d_rank_digit", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "d_rank_digit"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

<!-- #region kernel="SoS" -->
#### complexity model

- lag
- interaction_term
- quadratic_term
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
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                         total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
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
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "lag"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["interaction_term", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "interaction_term"})
                                .unstack(0)
                            ),
                            (
                                df.groupby(["quadratic_term", "target"])
                                .agg({"id_source": "nunique"})
                                .rename(columns={"id_source": "quadratic_term"})
                                .unstack(0)
                            ),
                        ],
                        axis=1,
                    )
                    .T.reset_index()
                    .rename(columns={"lag": "is_dummy", "level_0": "origin"})
                    .set_index(["origin", "is_dummy"])
                    .assign(
                        pct_significant=lambda x: x[("SIGNIFICANT")] / x.sum(axis=1),
                        total = lambda x: x['NOT_SIGNIFICANT'] + x['SIGNIFICANT'],
                        pct_total = lambda x: x['total']/x.groupby(['origin'])['total'].transform('sum')
                    )
                )
            ],
            axis=1,
            keys=["paper count"],
        ),
    ],
    axis=1,
).style.format("{:,.2%}", subset = [
    ('count','pct_significant'),
    ('count','pct_total'),
    ('paper count','pct_significant'),
    ('paper count','pct_total'),
])
```

```sos kernel="SoS"
df['interaction_term'].unique()
```

<!-- #region kernel="SoS" -->
#### nb_authors, pct_female, pct_esg_1
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("nb_authors")
                    .agg({"nb_authors": "count"})
                    .rename(columns={"nb_authors": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("nb_authors")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "nb_authors"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("nb_authors")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["nb_authors", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "nb_authors"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

```sos kernel="SoS"
(df.groupby("target").agg({"pct_female": "describe"}))
```

```sos kernel="SoS"
(df.groupby("target").agg({"pct_esg": "describe"}))
```

<!-- #region kernel="SoS" -->
#### sentiment, cluster_w_emb
<!-- #endregion -->

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("sentiment")
                    .agg({"sentiment": "count"})
                    .rename(columns={"sentiment": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("sentiment")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "sentiment"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("sentiment")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["sentiment", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "sentiment"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

```sos kernel="SoS"
pd.concat(
    [
        pd.concat(
            [
                (
                    df.groupby("cluster_w_emb")
                    .agg({"cluster_w_emb": "count"})
                    .rename(columns={"cluster_w_emb": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby("cluster_w_emb")
                    .agg({"target": "value_counts"})
                    .unstack(0)
                    .rename(columns={"target": "cluster_w_emb"})
                    .droplevel(axis=1, level=0)
                    .T
                )
            ],
            axis=1,
            keys=["count"],
        ),
         pd.concat(
            [
                (
                    df.groupby("cluster_w_emb")
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "count"})
                    .assign(pct=lambda x: x["count"] / np.sum(x["count"]))
                )
            ],
            axis=1,
            keys=["count paper raw"],
        ),
        pd.concat(
            [
                (
                    df.groupby(["cluster_w_emb", "target"])
                    .agg({"id_source": "nunique"})
                    .rename(columns={"id_source": "cluster_w_emb"})
                    .unstack(0)
                    .droplevel(axis=1, level=0)
                    .T
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
).style.format("{:,.2%}", subset = [
    ('count raw','pct'),
    ('count paper raw','pct'),
    ('paper count','pct_significant')
])
```

```sos kernel="SoS"
(
            df.groupby(["cluster_w_emb", "environmental"])
            .agg({"target": "value_counts"})
            .unstack(0)
     .rename(columns={"target": "environmental"})
    .T
        )
```

```sos kernel="SoS"
pd.concat(
    [
        (
            df.groupby(["cluster_w_emb", "environmental"])
            .agg({"target": "value_counts"})
            .unstack(0)
            .rename(columns={"target": "environmental"})
            .T
        ),
        (
            df.groupby(["cluster_w_emb", "social"])
            .agg({"target": "value_counts"})
            .unstack(0)
            .rename(columns={"target": "social"})
            .T
        ),
        (
            df.groupby(["cluster_w_emb", "governance"])
            .agg({"target": "value_counts"})
            .unstack(0)
            .rename(columns={"target": "governance"})
            .T
        ),
    ],
    axis=0,
)
```

<!-- #region kernel="SoS" -->
#### Correlation among covariates

- publication_year and mid_year are highly correlated, so cannot use them simultaneously
<!-- #endregion -->

```sos kernel="SoS"
# Compute the correlation matrix
corr = (
    df
    .reindex(columns = [
        'publication_year',
        'mid_year',
        "sjr",
        "windows",
        "nb_authors",
        "pct_female",
        "pct_esg"
    ])
    .corr()
)
(
    corr
    .where(np.triu(np.ones_like(corr, dtype=bool)))
    .T
    .style
    .format("{0:,.2f}",na_rep="-")
    #.background_gradient()
    #.applymap(lambda x: 'color: transparent' if pd.isnull(x) else '')
)
```

```sos kernel="SoS"
(
    df
    .reindex(columns = ['publication_year', 'mid_year'])
    .plot
    .scatter(x = 'mid_year', y = 'publication_year', figsize= (6,6) )
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

normalit<-function(m){
   (m - min(m))/(max(m)-min(m))
 }
df_final <- read_csv(df_path) %>%
mutate_if(is.character, as.factor) %>%

mutate(
    adjusted_model = relevel(adjusted_model, ref='OTHER'),
    adjusted_dependent = relevel(adjusted_dependent, ref='OTHER'),
    id_source = as.factor(id_source),
    governance = relevel(as.factor(governance), ref = 'NO'),
    social = relevel(as.factor(social), ref = 'NO'),
    environmental =relevel(as.factor(environmental), ref = 'NO'),
    financial_crisis =relevel(as.factor(financial_crisis), ref = 'NO'),
    kyoto =relevel(as.factor(kyoto), ref = 'NO'),
    target =relevel(as.factor(target), ref = 'NOT_SIGNIFICANT'),
    regions =relevel(as.factor(regions), ref = 'WORLDWIDE'),
    is_open_access =relevel(as.factor(is_open_access), ref = 'NO'),
    sentiment =relevel(as.factor(sentiment), ref = 'NEGATIVE'),
    region_journal =relevel(as.factor(region_journal), ref = 'NORTHERN AMERICA'),
    pct_esg_1 = normalit(pct_esg),
    esg_1 =  normalit(esg),
    sjr_1 =  normalit(sjr),
    cluster_w_emb = relevel(as.factor(cluster_w_emb), ref = 'CLUSTER_1'),
    citation_count_1 = normalit(citation_count),
    providers = relevel(as.factor(providers), ref = 'NOT_MSCI'),
    d_rank_digit = relevel(as.factor(d_rank_digit), ref = 'rank_2345')
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
- d_rang_digit: CNRS has 4 categories, ranging from 1 to 4. We added a 5th category for the missing one. The dummy compare the top journals (rank 1) vs. the others
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
- is_open_access: True if the journal is an open access publication
- region_journal: Region journal
    - Europe
        - Eastern Europe
        - Western Europe
    - Northern America
- providers: If `csr_20_categories` equals to 'MSCI'  then "MCSI" else "NOT_MSCI". MSCI is the main ESG's data provider
- nb_authors: Number of authors
- pct_female: Percentage of female authors
- pct_esg_1: ESG expert score calculated as the number of publications labeled as ESG over the total number of publications for all the authors of the paper
- Sentiment: Overall feeling of the abstract. Positive means the abstract tend to have more words associated with a positive connotation
- cluster_w_emb: 3 clusters computed using the words in the abstract (embeddings), the number of verbs, noun,s and adjectives but also the size of the abstract. 


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

<!-- #region kernel="R" -->
### Baseline table

The baseline regression accounts for: 

```
environmental # social governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year
           + windows
           + mid_year
           + regions
           + sjr
           + is_open_access
           + region_journal
           + providers
```

**Remove categorie**

Removing the categorie reduces the log-likelihood and reduce the AIC criteria (lower value, the better the model)

- adjusted_model:
    - DIFF IN DIFF
    - INSTRUMENT
    - LAG DEPENDENT
    - RANDOM EFFECT
- regions
     - LATIN AMERICA
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))
### Econometrics control
t_3 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

#### Remove low observations
t_6 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
),
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))
### Econometrics control
t_9 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit")
          )
t_9.rrr <- exp(coef(t_9))
t_10 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_10.rrr <- exp(coef(t_10))
t_11 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_11.rrr <- exp(coef(t_11))

list_final = list(t_0, t_1, t_2, t_3, t_4, t_5,t_6, t_7, t_8, t_9, t_10, t_11)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr,
                     t_6.rrr,t_7.rrr ,t_8.rrr,t_9.rrr,t_10.rrr,t_11.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="TABLES/table_0.txt"
         )
```

```sos kernel="R"
### Baseline CNRS
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final , binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final , binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))
### Econometrics control
t_3 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final , binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

#### Remove low observations
t_6 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
),
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))
### Econometrics control
t_9 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit")
          )
t_9.rrr <- exp(coef(t_9))
t_10 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_10.rrr <- exp(coef(t_10))
t_11 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA')
) ,
           binomial(link = "probit"))
t_11.rrr <- exp(coef(t_11))

list_final = list(t_0, t_1, t_2, t_3, t_4, t_5,t_6, t_7, t_8, t_9, t_10, t_11)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr,
                     t_6.rrr,t_7.rrr ,t_8.rrr,t_9.rrr,t_10.rrr,t_11.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="TABLES/table_1.txt"
         )
```

<!-- #region kernel="R" -->
#### Remove unknonw journals from CNRS
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(!rank_digit %in% c(5) &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')),
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(!rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(!rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))
### Econometrics control
t_3 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(!rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))
t_4 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(!rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit"))
t_4.rrr <- exp(coef(t_4))
t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(!rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) , 
           binomial(link = "probit"))
t_5.rrr <- exp(coef(t_5))

#### Remove low observations
t_6 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
),
           binomial(link = "probit")
          )
t_6.rrr <- exp(coef(t_6))
t_7 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
),
           binomial(link = "probit"))
t_7.rrr <- exp(coef(t_7))
t_8 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
),
           binomial(link = "probit"))
t_8.rrr <- exp(coef(t_8))
### Econometrics control
t_9 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
) ,
           binomial(link = "probit")
          )
t_9.rrr <- exp(coef(t_9))
t_10 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
) ,
           binomial(link = "probit"))
t_10.rrr <- exp(coef(t_10))
t_11 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + lag
           + interaction_term,
           data = df_final %>% filter(
    !adjusted_model %in%  c('DIFF IN DIFF', 'INSTRUMENT', 'LAG DEPENDENT', 'RANDOM EFFECT')
    &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')
               &
               !rank_digit %in% c(5)
) ,
           binomial(link = "probit"))
t_11.rrr <- exp(coef(t_11))

list_final = list(t_0, t_1, t_2, t_3, t_4, t_5,t_6, t_7, t_8, t_9, t_10, t_11)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr,
                     t_6.rrr,t_7.rrr ,t_8.rrr,t_9.rrr,t_10.rrr,t_11.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje"
         #out="TABLES/table_0.txt"
         )
```

<!-- #region kernel="R" -->
#### Keep unknow journals
<!-- #endregion -->

```sos kernel="R"
### Baseline SJR
t_0 <- glm(target ~ environmental
           + adjusted_model  
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           #+ region_journal
           + providers
           + sjr,
           data = df_final %>% filter(rank_digit %in% c(5) &
    !regions %in%  c('LATIN AMERICA', 'AFRICA')),
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))
t_1 <- glm(target ~ social
           + adjusted_model    
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           #+ region_journal
           + providers
           + sjr,
           data = df_final %>% filter(rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit"))
t_1.rrr <- exp(coef(t_1))
t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           #+ financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           #+ region_journal
           + providers
           + sjr,
           data = df_final %>% filter(rank_digit %in% c(5)&
    !regions %in%  c('LATIN AMERICA', 'AFRICA')) ,
           binomial(link = "probit"))
t_2.rrr <- exp(coef(t_2))

list_final = list(t_0, t_1, t_2)
list_final.rrr = list(t_0.rrr,t_1.rrr ,t_2.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje"
         #out="TABLES/table_0.txt"
         )
```

<!-- #region kernel="R" -->
### Papers and authors specification


In the second tables, we focus on the authors informations:

Baseline variables + 

- nb_authors: Number of authors in the paper
- pct_female: Percentage of female author
- pct_esg_1: ESG expertise of the authors (normalized value)


<!-- #endregion -->

```sos kernel="R"
###
t_0 <- glm(target ~ environmental
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_2.rrr <- exp(coef(t_2))

### CNRS
t_3 <- glm(target ~ environmental
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))


t_4 <- glm(target ~ social
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_4.rrr <- exp(coef(t_4))

t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + nb_authors
           + pct_female
           + pct_esg_1,
           data = df_final ,
           binomial(link = "probit")
          )
t_5.rrr <- exp(coef(t_5))

list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="TABLES/table_2.txt")
```

<!-- #region kernel="R" -->
### Characteristic abstract

**Sentiment**: Overall feeling of the abstract. Positive means the abstract tend to have more words associated with a positive connotation

**cluster_w_emb**: 3 clusters computed using the words in the abstract (embeddings), the number of verbs, noun,s and adjectives but also the size of the abstract. 

The k-mean algorithm clustered the abstract based on the "quality" of it. 

- sentiment
- cluster

![](https://storage.googleapis.com/memvp-25499.appspot.com/images/image.png76221423-1aa4-4af3-a9e0-3a6b46b0c1f2)
<!-- #endregion -->

```sos kernel="R"
###
t_0 <- glm(target ~ environmental
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_0.rrr <- exp(coef(t_0))


t_1 <- glm(target ~ social
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_1.rrr <- exp(coef(t_1))

t_2 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_2.rrr <- exp(coef(t_2))

### CNRS
t_3 <- glm(target ~ environmental
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_3.rrr <- exp(coef(t_3))


t_4 <- glm(target ~ social
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_4.rrr <- exp(coef(t_4))

t_5 <- glm(target ~ governance
           + adjusted_model
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + sentiment
           + cluster_w_emb,
           data = df_final ,
           binomial(link = "probit")
          )
t_5.rrr <- exp(coef(t_5))

list_final = list(t_0, t_1, t_2,t_3, t_4, t_5)
list_final.rrr = list(t_0.rrr,t_1.rrr,t_2.rrr,t_3.rrr,t_4.rrr,t_5.rrr)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          coef=list_final.rrr,
          style = "qje",
         out="TABLES/table_3.txt")
```

<!-- #region kernel="SoS" heading_collapsed="true" -->
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
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
           + publication_year_int
           + windows
           + mid_year
           + regions
           + sjr
           + is_open_access
           + region_journal
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

### Econometrics control
t_3 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_4 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
           + lag
           + interaction_term
           + quadratic_term,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity))

t_5 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rank_digit
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
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

```sos kernel="R"
###
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
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
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
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
Abstract
<!-- #endregion -->

```sos kernel="R"
###
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )
t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + sjr
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

list_final = list(t_0, t_1, t_2)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
```

```sos kernel="R"
###
t_0 <- glm(adjusted_t_value ~ environmental
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )
t_1 <- glm(adjusted_t_value ~ social
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

t_2 <- glm(adjusted_t_value ~ governance
           + adjusted_model  
           + kyoto 
           + financial_crisis
           + publication_year_int
           + windows
           #+ mid_year
           + regions
           + is_open_access
           + region_journal
           + providers
           + d_rang_digit
           + sentiment
           + cluster_w_emb,
           data = df_final %>% filter(adjusted_t_value < 10),
           family=gaussian(identity)
          )

list_final = list(t_0, t_1, t_2)
stargazer(list_final, type = "text", 
  se = lapply(list_final,
              se_robust),
          style = "qje")
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
