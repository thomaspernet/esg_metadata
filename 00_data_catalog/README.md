
# Data Catalogue



## Table of Content

    
- [journals_scimago](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog#table-journals_scimago)
- [papers_meta_analysis_new](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog#table-papers_meta_analysis_new)
- [meta_analysis_esg_cfp](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog#table-meta_analysis_esg_cfp)

    

## Table journals_scimago

- Database: scimago
- S3uri: `s3://datalake-london/DATA/JOURNALS/SCIMAGO`
- Partitition: []
- Script: https://github.com/thomaspernet/esg_metadata/01_data_preprocessing/00_download_data/SCIMAGO/scimagojr.py

|    | Name                | Type   | Comment                                                                                                                                 |
|---:|:--------------------|:-------|:----------------------------------------------------------------------------------------------------------------------------------------|
|  0 | rank                | bigint | journal rank                                                                                                                            |
|  1 | sourceid            | bigint | source                                                                                                                                  |
|  2 | title               | string | Journal title                                                                                                                           |
|  3 | type                | string | Type of journal                                                                                                                         |
|  4 | issn                | string | ISSN                                                                                                                                    |
|  5 | sjr                 | float  | SCImago Journal Rank                                                                                                                    |
|  6 | sjr_best_quartile   | string | SCImago Journal Rank rank by quartile                                                                                                   |
|  7 | h_index             | float  | h-index is an author-level metric that measures both the productivity and citation impact of the publications of a scientist or scholar |
|  8 | total_docs          | float  | total doc in 2020                                                                                                                       |
|  9 | total_docs_3years   | float  | total doc past 3 years                                                                                                                  |
| 10 | total_refs          | float  | total references                                                                                                                        |
| 11 | total_cites_3years  | float  | total cited last 3 years                                                                                                                |
| 12 | citable_docs_3years | float  | citable doc                                                                                                                             |
| 13 | cites_doc_2years    | float  | citation per doc over the last 3 years                                                                                                  |
| 14 | ref_doc             | float  | number of reference per doc                                                                                                             |
| 15 | country             | string | country of origin                                                                                                                       |
| 16 | region              | string | region of origin                                                                                                                        |
| 17 | publisher           | string | publisher                                                                                                                               |
| 18 | coverage            | string | coverage                                                                                                                                |
| 19 | categories          | string | categories                                                                                                                              |

    

## Table papers_meta_analysis_new

- Database: esg
- S3uri: `s3://datalake-london/DATA/FINANCE/ESG/META_ANALYSIS_NEW`
- Partitition: []
- Script: https://github.com/thomaspernet/esg_metadata/01_data_preprocessing/00_download_data/METADATA_TABLES/esg_metaanalysis.py

|    | Name                 | Type   | Comment                                                                               |
|---:|:---------------------|:-------|:--------------------------------------------------------------------------------------|
|  0 | id                   | string | paper ID                                                                              |
|  1 | paper_name           | string | Paper name                                                                            |
|  2 | dependent            | string | dependent variable                                                                    |
|  3 | independent          | string | independent variables                                                                 |
|  4 | lag                  | string | the table contains lag or not                                                         |
|  5 | interaction_term     | string | the table contains interaction terms or not                                           |
|  6 | quadratic_term       | string | the table contains quadratic terms or not                                             |
|  7 | n                    | float  | number of observations                                                                |
|  8 | r2                   | float  | R square                                                                              |
|  9 | beta                 | float  | Beta coefficient                                                                      |
| 10 | sign                 | string | sign positive or negative                                                             |
| 11 | star                 | string | Level of significant. *, ** or ***                                                    |
| 12 | sr                   | float  | standard error                                                                        |
| 13 | p_value              | float  | p value                                                                               |
| 14 | t_value              | float  | student value                                                                         |
| 15 | table_refer          | string | table number in the paper                                                             |
| 16 | model_name           | string | Model name from Susie                                                                 |
| 17 | updated_model_name   | string | Model name adjusted by Thomas                                                         |
| 18 | adjusted_model_name  | string | Model name normalised                                                                 |
| 19 | doi                  | string | DOI                                                                                   |
| 20 | drive_url            | string | paper link in Google Drive                                                            |
| 21 | test_standard_error  | string | check if sr really standard error by comparing beta divided by sr and critical values |
| 22 | test_p_value         | string | check if sign and p value match                                                       |
| 23 | test_t_value         | string | check if t critial value and sign match                                               |
| 24 | should_t_value       | string | use sr as t value when mistake                                                        |
| 25 | true_standard_error  | float  | reconstructed standard error                                                          |
| 26 | true_t_value         | float  | reconstructed t value                                                                 |
| 27 | true_stars           | string | reconstructed stars                                                                   |
| 28 | adjusted_dependent   | string | reorganise dependent variable into smaller groups                                     |
| 29 | adjusted_independent | float  | reorganise independent variable into smaller group                                    |
| 30 | adjusted_model       | string | reorganise model variable into smaller group                                          |
| 31 | significant          | string | is beta significant                                                                   |
| 32 | to_check_final       | string | Final check rows                                                                      |
| 33 | critical_99          | string | 99 t stat critical value calculated based on nb obserations                           |
| 34 | critical_90          | string | 90 t stat critical value calculated based on nb obserations                           |
| 35 | critical_95          | string | 95 t stat critical value calculated based on nb obserations                           |

    

## Table meta_analysis_esg_cfp

- Database: esg
- S3uri: `s3://datalake-london/DATA/FINANCE/ESG/ESG_CFP`
- Partitition: ['id', 'incremental_id']
- Script: https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/01_transform_tables/00_meta_analysis.md

|    | Name                                                | Type   | Comment                                                                                                                                 |
|---:|:----------------------------------------------------|:-------|:----------------------------------------------------------------------------------------------------------------------------------------|
|  0 | id                                                  | string | paper ID                                                                                                                                |
|  1 | incremental_id                                      | string | row id                                                                                                                                  |
|  2 | paper_name                                          | string | Paper name                                                                                                                              |
|  3 | publication_year                                    | float  | publication year                                                                                                                        |
|  4 | publication_type                                    | string | publication type                                                                                                                        |
|  5 | publication_name                                    | string | publication name                                                                                                                        |
|  6 | cnrs_ranking                                        | string | cnrs ranking                                                                                                                            |
|  7 | peer_reviewed                                       | string | peer reviewed                                                                                                                           |
|  8 | study_focused_on_social_environmental_behaviour     | string | study focused on social environmental behaviour                                                                                         |
|  9 | type_of_data                                        | string | type of data                                                                                                                            |
| 10 | study_focusing_on_developing_or_developed_countries | string | study focusing on developing or developed countries                                                                                     |
| 11 | first_date_of_observations                          | int    | first date of observations                                                                                                              |
| 12 | last_date_of_observations                           | int    | last date of observations                                                                                                               |
| 13 | dependent                                           | string | dependent variable                                                                                                                      |
| 14 | independent                                         | string | independent variables                                                                                                                   |
| 15 | lag                                                 | string | the table contains lag or not                                                                                                           |
| 16 | interaction_term                                    | string | the table contains interaction terms or not                                                                                             |
| 17 | quadratic_term                                      | string | the table contains quadratic terms or not                                                                                               |
| 18 | n                                                   | float  | number of observations                                                                                                                  |
| 19 | r2                                                  | float  | R square                                                                                                                                |
| 20 | beta                                                | float  | Beta coefficient                                                                                                                        |
| 21 | to_remove                                           | string | to remove                                                                                                                               |
| 22 | critical_value                                      | double | critical value                                                                                                                          |
| 23 | true_standard_error                                 | float  | reconstructed standard error                                                                                                            |
| 24 | true_t_value                                        | float  | reconstructed t value                                                                                                                   |
| 25 | true_stars                                          | string | reconstructed stars                                                                                                                     |
| 26 | adjusted_dependent                                  | string | reorganise dependent variable into smaller groups                                                                                       |
| 27 | adjusted_independent                                | float  | reorganise independent variable into smaller group                                                                                      |
| 28 | adjusted_model                                      | string | reorganise model variable into smaller group                                                                                            |
| 29 | significant                                         | string | is beta significant                                                                                                                     |
| 30 | to_check_final                                      | string | Final check rows                                                                                                                        |
| 31 | rank                                                | bigint | journal rank                                                                                                                            |
| 32 | title                                               | string | title                                                                                                                                   |
| 33 | sjr                                                 | float  | SCImago Journal Rank                                                                                                                    |
| 34 | sjr_best_quartile                                   | string | SCImago Journal Rank rank by quartile                                                                                                   |
| 35 | h_index                                             | float  | h-index is an author-level metric that measures both the productivity and citation impact of the publications of a scientist or scholar |
| 36 | total_docs_2020                                     | bigint | total docs 2020                                                                                                                         |
| 37 | total_docs_3years                                   | float  | total doc past 3 years                                                                                                                  |
| 38 | total_refs                                          | float  | total references                                                                                                                        |
| 39 | total_cites_3years                                  | float  | total cited last 3 years                                                                                                                |
| 40 | citable_docs_3years                                 | float  | citable doc                                                                                                                             |
| 41 | cites_doc_2years                                    | float  | citation per doc over the last 3 years                                                                                                  |
| 42 | country                                             | string | country of origin                                                                                                                       |

    