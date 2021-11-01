
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

|    | Name                       | Type    | Comment                                                                                                     |
|---:|:---------------------------|:--------|:------------------------------------------------------------------------------------------------------------|
|  0 | paperid                    | string  | paperid                                                                                                     |
|  1 | nb_authors                 | int     | nb authors                                                                                                  |
|  2 | reference_count            | int     | reference count                                                                                             |
|  3 | citation_count             | int     | citation count                                                                                              |
|  4 | influential_citation_count | int     | influential citation count                                                                                  |
|  5 | is_open_access             | boolean | is open access                                                                                              |
|  6 | total_paper                | int     | total paper                                                                                                 |
|  7 | esg                        | int     | esg                                                                                                         |
|  8 | pct_esg                    | float   | pct esg                                                                                                     |
|  9 | id_source                  | string  | id source                                                                                                   |
| 10 | female                     | int     | female                                                                                                      |
| 11 | male                       | int     | male                                                                                                        |
| 12 | unknown                    | int     | unknown                                                                                                     |
| 13 | pct_female                 | float   | pct female                                                                                                  |
| 14 | drive_url                  | string  | paper link in Google Drive                                                                                  |
| 15 | image                      | string  | Link row data image                                                                                         |
| 16 | row_id_google_spreadsheet  | string  | Google spreadsheet link to raw data                                                                         |
| 17 | table_refer                | string  | table number in the paper                                                                                   |
| 18 | adjusted_model             | string  | reorganise model variable into smaller group                                                                |
| 19 | adjusted_dependent         | string  | reorganise dependent variable into smaller groups                                                           |
| 20 | adjusted_independent       | string  | reorganise independent variable into smaller group                                                          |
| 21 | social                     | string  | if adjusted_independent in ENVIRONMENTAL AND SOCIAL, SOCIAL, CSP, CSR, ENVIRONMENTAL, SOCIAL and GOVERNANCE |
| 22 | environmental              | string  | if adjusted_independent in ENVIRONMENTAL, ENVIRONMENTAL AND SOCIAL, ENVIRONMENTAL, SOCIAL and GOVERNANCE    |
| 23 | governance                 | string  | if adjusted_independent in GOVERNANCE ENVIRONMENTAL, SOCIAL and GOVERNANCE                                  |
| 24 | lag                        | string  | the table contains lag or not                                                                               |
| 25 | interaction_term           | string  | the table contains interaction terms or not                                                                 |
| 26 | quadratic_term             | string  | the table contains quadratic terms or not                                                                   |
| 27 | n                          | int     | number of observations                                                                                      |
| 28 | target                     | string  | indicate wheither or not the coefficient is significant. based on stars                                     |
| 29 | adjusted_standard_error    | float   | reconstructed standard error                                                                                |
| 30 | adjusted_t_value           | float   | reconstructed t value                                                                                       |
| 31 | paper_name                 | string  | Paper name                                                                                                  |
| 32 | first_date_of_observations | int     | first date of observations                                                                                  |
| 33 | last_date_of_observations  | int     | last date of observations                                                                                   |
| 34 | csr_20_categories          | string  | csr 20 categories                                                                                           |
| 35 | kyoto                      | string  | kyoto                                                                                                       |
| 36 | financial_crisis           | string  | financial crisis                                                                                            |
| 37 | windows                    | int     | windows                                                                                                     |
| 38 | mid_year                   | float   | mid year                                                                                                    |
| 39 | regions                    | string  | regions                                                                                                     |
| 40 | providers                  | string  | providers                                                                                                   |
| 41 | publication_year           | int     | publication year                                                                                            |
| 42 | publication_name           | string  | publication name                                                                                            |
| 43 | rank_digit                 | string  | rank digit                                                                                                  |
| 44 | cluster_w_emb              | int     | cluster w emb                                                                                               |
| 45 | sentiment                  | string  | sentiment                                                                                                   |
| 46 | lenght                     | int     | lenght                                                                                                      |
| 47 | adj                        | int     | adj                                                                                                         |
| 48 | noun                       | int     | noun                                                                                                        |
| 49 | verb                       | int     | verb                                                                                                        |
| 50 | size_abstract              | int     | size abstract                                                                                               |
| 51 | pct_adj                    | float   | pct adj                                                                                                     |
| 52 | pct_noun                   | float   | pct noun                                                                                                    |
| 53 | pct_verb                   | float   | pct verb                                                                                                    |
| 54 | rank                       | int     | rank                                                                                                        |
| 55 | sjr                        | float   | sjr                                                                                                         |
| 56 | region_journal             | string  | region journal                                                                                              |

    