
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

|    | Name                      | Type   | Comment                                                                                                                 |
|---:|:--------------------------|:-------|:------------------------------------------------------------------------------------------------------------------------|
|  0 | id                        | string | paper ID                                                                                                                |
|  1 | incremental_id            | string | row id                                                                                                                  |
|  2 | paper_name                | string | Paper name                                                                                                              |
|  3 | dependent                 | string | dependent variable                                                                                                      |
|  4 | independent               | string | independent variables                                                                                                   |
|  5 | lag                       | string | the table contains lag or not                                                                                           |
|  6 | interaction_term          | string | the table contains interaction terms or not                                                                             |
|  7 | quadratic_term            | string | the table contains quadratic terms or not                                                                               |
|  8 | n                         | float  | number of observations                                                                                                  |
|  9 | r2                        | float  | R square                                                                                                                |
| 10 | beta                      | float  | Beta coefficient                                                                                                        |
| 11 | sign                      | string | sign positive or negative                                                                                               |
| 12 | star                      | string | Level of significant. *, ** or ***                                                                                      |
| 13 | sr                        | float  | standard error                                                                                                          |
| 14 | p_value                   | float  | p value                                                                                                                 |
| 15 | t_value                   | float  | student value                                                                                                           |
| 16 | image                     | string | Link row data image                                                                                                     |
| 17 | table_refer               | string | table number in the paper                                                                                               |
| 18 | model_name                | string | Model name from Susie                                                                                                   |
| 19 | updated_model_name        | string | Model name adjusted by Thomas                                                                                           |
| 20 | adjusted_model_name       | string | Model name normalised                                                                                                   |
| 21 | doi                       | string | DOI                                                                                                                     |
| 22 | drive_url                 | string | paper link in Google Drive                                                                                              |
| 23 | test_standard_error       | string | check if sr really standard error by comparing beta divided by sr and critical values                                   |
| 24 | test_p_value              | string | check if sign and p value match                                                                                         |
| 25 | test_t_value              | string | check if t critial value and sign match                                                                                 |
| 26 | should_t_value            | string | use sr as t value when mistake                                                                                          |
| 27 | adjusted_standard_error   | float  | reconstructed standard error                                                                                            |
| 28 | final_standard_error      | float  | reconstructed standard error and use sr when true_standard_error is nan or error                                        |
| 29 | adjusted_t_value          | float  | reconstructed t value                                                                                                   |
| 30 | true_stars                | string | reconstructed stars                                                                                                     |
| 31 | adjusted_dependent        | string | reorganise dependent variable into smaller groups                                                                       |
| 32 | adjusted_independent      | string | reorganise independent variable into smaller group                                                                      |
| 33 | adjusted_model            | string | reorganise model variable into smaller group                                                                            |
| 34 | p_value_significant       | string | is beta significant. Computed from reconstructed critical values, not the paper. From paper, see variable target        |
| 35 | to_check_final            | string | Final check rows                                                                                                        |
| 36 | critical_99               | string | 99 t stat critical value calculated based on nb obserations                                                             |
| 37 | critical_90               | string | 90 t stat critical value calculated based on nb obserations                                                             |
| 38 | critical_95               | string | 95 t stat critical value calculated based on nb obserations                                                             |
| 39 | social                    | string | if adjusted_independent in ENVIRONMENTAL AND SOCIAL, SOCIAL, CSP, CSR, ENVIRONMENTAL, SOCIAL and GOVERNANCE             |
| 40 | environmental             | string | if adjusted_independent in ENVIRONMENTAL, ENVIRONMENTAL AND SOCIAL, ENVIRONMENTAL, SOCIAL and GOVERNANCE                |
| 41 | governance                | string | if adjusted_independent in GOVERNANCE ENVIRONMENTAL, SOCIAL and GOVERNANCE                                              |
| 42 | sign_of_effect            | string | if stars is not blank and beta > 0, then POSITIVE, if stars is not blank and beta < 0, then NEGATIVE else INSIGNIFICANT |
| 43 | row_id_google_spreadsheet | string | Google spreadsheet link to raw data                                                                                     |
| 44 | sign_positive             | string | if sign_of_effect is POSITIVE then True                                                                                 |
| 45 | sign_negative             | string | if sign_of_effect is NEGATIVE then True                                                                                 |
| 46 | sign_significant          | string | if sign_of_effect is INSIGNIFICANT then False else True                                                                 |
| 47 | model_instrument          | string | if adjusted_model is equal to INSTRUMENT then true                                                                      |
| 48 | model_diff_in_diff        | string | if adjusted_model is equal to DIFF IN DIFF then true                                                                    |
| 49 | model_other               | string | if adjusted_model is equal to OTHER then true                                                                           |
| 50 | model_fixed_effect        | string | if adjusted_model is equal to FIXED EFFECT then true                                                                    |
| 51 | model_lag_dependent       | string | if adjusted_model is equal to LAG DEPENDENT then true                                                                   |
| 52 | model_pooled_ols          | string | if adjusted_model is equal to POOLED OLS then true                                                                      |
| 53 | model_random_effect       | string | if adjusted_model is equal to RANDOM EFFECT then true                                                                   |
| 54 | target                    | string | indicate wheither or not the coefficient is significant. based on stars                                                 |

    

## Table meta_analysis_esg_cfp

- Database: esg
- S3uri: `s3://datalake-london/DATA/FINANCE/ESG/ESG_CFP`
- Partitition: ['id', 'incremental_id']
- Script: https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/01_transform_tables/00_meta_analysis.md

|    | Name                                                | Type        | Comment                                                                                                                                 |
|---:|:----------------------------------------------------|:------------|:----------------------------------------------------------------------------------------------------------------------------------------|
|  0 | to_remove                                           | string      | to remove                                                                                                                               |
|  1 | id                                                  | string      | paper ID                                                                                                                                |
|  2 | image                                               | string      | Link row data image                                                                                                                     |
|  3 | row_id_excel                                        | string      | link to original row                                                                                                                    |
|  4 | row_id_google_spreadsheet                           | string      | Google spreadsheet link to raw data                                                                                                     |
|  5 | table_refer                                         | string      | table number in the paper                                                                                                               |
|  6 | incremental_id                                      | string      | row id                                                                                                                                  |
|  7 | paper_name                                          | string      | Paper name                                                                                                                              |
|  8 | publication_name                                    | string      | publication name                                                                                                                        |
|  9 | rank                                                | bigint      | journal rank                                                                                                                            |
| 10 | sjr                                                 | float       | SCImago Journal Rank                                                                                                                    |
| 11 | sjr_best_quartile                                   | string      | SCImago Journal Rank rank by quartile                                                                                                   |
| 12 | h_index                                             | float       | h-index is an author-level metric that measures both the productivity and citation impact of the publications of a scientist or scholar |
| 13 | total_docs_2020                                     | bigint      | total docs 2020                                                                                                                         |
| 14 | total_docs_3years                                   | float       | total doc past 3 years                                                                                                                  |
| 15 | total_refs                                          | float       | total references                                                                                                                        |
| 16 | total_cites_3years                                  | float       | total cited last 3 years                                                                                                                |
| 17 | citable_docs_3years                                 | float       | citable doc                                                                                                                             |
| 18 | cites_doc_2years                                    | float       | citation per doc over the last 3 years                                                                                                  |
| 19 | country                                             | string      | country of origin                                                                                                                       |
| 20 | publication_year                                    | float       | publication year                                                                                                                        |
| 21 | publication_type                                    | string      | publication type                                                                                                                        |
| 22 | cnrs_ranking                                        | string      | cnrs ranking                                                                                                                            |
| 23 | peer_reviewed                                       | string      | peer reviewed                                                                                                                           |
| 24 | study_focused_on_social_environmental_behaviour     | string      | study focused on social environmental behaviour                                                                                         |
| 25 | type_of_data                                        | string      | type of data                                                                                                                            |
| 26 | regions                                             | varchar(16) | regions                                                                                                                                 |
| 27 | study_focusing_on_developing_or_developed_countries | string      | study focusing on developing or developed countries                                                                                     |
| 28 | first_date_of_observations                          | int         | first date of observations                                                                                                              |
| 29 | last_date_of_observations                           | int         | last date of observations                                                                                                               |
| 30 | kyoto                                               | varchar(3)  | kyoto                                                                                                                                   |
| 31 | financial_crisis                                    | varchar(3)  | financial crisis                                                                                                                        |
| 32 | windows                                             | int         | windows                                                                                                                                 |
| 33 | adjusted_model_name                                 | string      | Model name normalised                                                                                                                   |
| 34 | adjusted_model                                      | string      | reorganise model variable into smaller group                                                                                            |
| 35 | dependent                                           | string      | dependent variable                                                                                                                      |
| 36 | adjusted_dependent                                  | string      | reorganise dependent variable into smaller groups                                                                                       |
| 37 | independent                                         | string      | independent variables                                                                                                                   |
| 38 | adjusted_independent                                | string      | reorganise independent variable into smaller group                                                                                      |
| 39 | social                                              | string      | if adjusted_independent in ENVIRONMENTAL AND SOCIAL, SOCIAL, CSP, CSR, ENVIRONMENTAL, SOCIAL and GOVERNANCE                             |
| 40 | environmental                                       | string      | if adjusted_independent in ENVIRONMENTAL, ENVIRONMENTAL AND SOCIAL, ENVIRONMENTAL, SOCIAL and GOVERNANCE                                |
| 41 | governance                                          | string      | if adjusted_independent in GOVERNANCE ENVIRONMENTAL, SOCIAL and GOVERNANCE                                                              |
| 42 | sign_of_effect                                      | string      | if stars is not blank and beta > 0, then POSITIVE, if stars is not blank and beta < 0, then NEGATIVE else INSIGNIFICANT                 |
| 43 | target                                              | string      | indicate wheither or not the coefficient is significant. based on stars                                                                 |
| 44 | p_value_significant                                 | string      | is beta significant. Computed from reconstructed critical values, not the paper. From paper, see variable target                        |
| 45 | sign_positive                                       | string      | if sign_of_effect is POSITIVE then True                                                                                                 |
| 46 | sign_negative                                       | string      | if sign_of_effect is NEGATIVE then True                                                                                                 |
| 47 | lag                                                 | string      | the table contains lag or not                                                                                                           |
| 48 | interaction_term                                    | string      | the table contains interaction terms or not                                                                                             |
| 49 | quadratic_term                                      | string      | the table contains quadratic terms or not                                                                                               |
| 50 | n                                                   | float       | number of observations                                                                                                                  |
| 51 | r2                                                  | float       | R square                                                                                                                                |
| 52 | beta                                                | float       | Beta coefficient                                                                                                                        |
| 53 | test_standard_error                                 | string      | check if sr really standard error by comparing beta divided by sr and critical values                                                   |
| 54 | test_p_value                                        | string      | check if sign and p value match                                                                                                         |
| 55 | test_t_value                                        | string      | check if t critial value and sign match                                                                                                 |
| 56 | adjusted_standard_error                             | float       | reconstructed standard error                                                                                                            |
| 57 | adjusted_t_value                                    | float       | reconstructed t value                                                                                                                   |

    