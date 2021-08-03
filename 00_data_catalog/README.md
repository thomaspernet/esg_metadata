
# Data Catalogue



## Table of Content

    
- [journals_scimago](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog#table-journals_scimago)
- [papers_meta_analysis](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog#table-papers_meta_analysis)
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

    

## Table papers_meta_analysis

- Database: esg
- S3uri: `s3://datalake-london/DATA/FINANCE/ESG/META_ANALYSIS`
- Partitition: []
- Script: https://github.com/thomaspernet/esg_metadata/01_data_preprocessing/00_download_data/ESG/esg_metaanalysis.py

|    | Name                                                | Type   | Comment              |
|---:|:----------------------------------------------------|:-------|:---------------------|
|  0 | nr                                                  | string |                      |
|  1 | title                                               | string |                      |
|  2 | first_author                                        | string |                      |
|  3 | second_author                                       | string |                      |
|  4 | third_author                                        | string |                      |
|  5 | publication_year                                    | float  |                      |
|  6 | publication_type                                    | string |                      |
|  7 | publication_name                                    | string |                      |
|  8 | cnrs_ranking                                        | string |                      |
|  9 | ranking                                             | float  |                      |
| 10 | peer_reviewed                                       | string |                      |
| 11 | study_focused_on_social_environmental_behaviour     | string |                      |
| 12 | comments_on_sample                                  | string |                      |
| 13 | type_of_data                                        | string |                      |
| 14 | sample_size_number_of_companies                     | float  |                      |
| 15 | first_date_of_observations                          | float  |                      |
| 16 | last_date_of_observations                           | float  |                      |
| 17 | number_of_observations                              | float  |                      |
| 18 | regions_of_selected_firms                           | string |                      |
| 19 | study_focusing_on_developing_or_developed_countries | string |                      |
| 20 | measure_of_corporate_social_responsibility_crp      | string |                      |
| 21 | csr_7_categories                                    | string |                      |
| 22 | csr_20_categories                                   | string |                      |
| 23 | unit_for_measure_of_environmental_behaviour         | string |                      |
| 24 | measure_of_financial_performance                    | string |                      |
| 25 | cfp_26_categories                                   | string |                      |
| 26 | unit_for_measure_of_financial_performance           | string |                      |
| 27 | cfp_4_categories                                    | string |                      |
| 28 | lagged_csr_explanatory_variable                     | string |                      |
| 29 | evaluation_method_of_the_link_between_csr_and_cfp   | string |                      |
| 30 | developed_new                                       | string |                      |
| 31 | definition_of_cfp_as_dependent_variable             | string |                      |
| 32 | comments                                            | string |                      |
| 33 | cfp_regrouping                                      | string |                      |
| 34 | level_of_significancy                               | float  |                      |
| 35 | sign_of_effect                                      | string |                      |
| 36 | standard_error                                      | float  |                      |
| 37 | tstatistic_calculated_with_formula                  | float  |                      |
| 38 | pvalue_calculated_with_formula                      | float  |                      |
| 39 | effect_coeffient_estimator_beta                     | float  |                      |
| 40 | adjusted_coefficient_of_determination               | float  |                      |
| 41 | econometric_method                                  | string |                      |
| 42 | kyoto_db                                            | string |                      |
| 43 | crisis_db                                           | string |                      |
| 44 | row_id_excel                                        | string | link to original row |

    

## Table meta_analysis_esg_cfp

- Database: esg
- S3uri: `s3://datalake-london/DATA/FINANCE/ESG/ESG_CFP`
- Partitition: ['id', 'incremental_id']
- Script: https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/01_transform_tables/00_meta_analysis.md

|    | Name                                                | Type   | Comment                                                                                                                                 |
|---:|:----------------------------------------------------|:-------|:----------------------------------------------------------------------------------------------------------------------------------------|
|  0 | to_remove                                           | string | to remove                                                                                                                               |
|  1 | id                                                  | string | paper ID                                                                                                                                |
|  2 | url_image                                           | string | url image                                                                                                                               |
|  3 | url_excel                                           | string | url excel                                                                                                                               |
|  4 | url_google_spreadsheet                              | string | url google spreadsheet                                                                                                                  |
|  5 | drive_url                                           | string | paper link in Google Drive                                                                                                              |
|  6 | table_refer                                         | string | table number in the paper                                                                                                               |
|  7 | incremental_id                                      | string | row id                                                                                                                                  |
|  8 | paper_name                                          | string | Paper name                                                                                                                              |
|  9 | publication_name                                    | string | publication name                                                                                                                        |
| 10 | rank                                                | bigint | journal rank                                                                                                                            |
| 11 | sjr                                                 | float  | SCImago Journal Rank                                                                                                                    |
| 12 | sjr_best_quartile                                   | string | SCImago Journal Rank rank by quartile                                                                                                   |
| 13 | h_index                                             | float  | h-index is an author-level metric that measures both the productivity and citation impact of the publications of a scientist or scholar |
| 14 | total_docs_2020                                     | bigint | total docs 2020                                                                                                                         |
| 15 | total_docs_3years                                   | float  | total doc past 3 years                                                                                                                  |
| 16 | total_refs                                          | float  | total references                                                                                                                        |
| 17 | total_cites_3years                                  | float  | total cited last 3 years                                                                                                                |
| 18 | citable_docs_3years                                 | float  | citable doc                                                                                                                             |
| 19 | cites_doc_2years                                    | float  | citation per doc over the last 3 years                                                                                                  |
| 20 | country                                             | string | country of origin                                                                                                                       |
| 21 | publication_year                                    | float  | publication year                                                                                                                        |
| 22 | publication_type                                    | string | publication type                                                                                                                        |
| 23 | cnrs_ranking                                        | string | cnrs ranking                                                                                                                            |
| 24 | peer_reviewed                                       | string | peer reviewed                                                                                                                           |
| 25 | study_focused_on_social_environmental_behaviour     | string | study focused on social environmental behaviour                                                                                         |
| 26 | type_of_data                                        | string | type of data                                                                                                                            |
| 27 | study_focusing_on_developing_or_developed_countries | string | study focusing on developing or developed countries                                                                                     |
| 28 | first_date_of_observations                          | int    | first date of observations                                                                                                              |
| 29 | last_date_of_observations                           | int    | last date of observations                                                                                                               |
| 30 | adjusted_model                                      | string | reorganise model variable into smaller group                                                                                            |
| 31 | model_instrument                                    | string | if adjusted_model is equal to INSTRUMENT then true                                                                                      |
| 32 | model_diff_in_diff                                  | string | if adjusted_model is equal to DIFF IN DIFF then true                                                                                    |
| 33 | model_other                                         | string | if adjusted_model is equal to OTHER then true                                                                                           |
| 34 | model_fixed_effect                                  | string | if adjusted_model is equal to FIXED EFFECT then true                                                                                    |
| 35 | model_lag_dependent                                 | string | if adjusted_model is equal to LAG DEPENDENT then true                                                                                   |
| 36 | model_pooled_ols                                    | string | if adjusted_model is equal to POOLED OLS then true                                                                                      |
| 37 | model_random_effect                                 | string | if adjusted_model is equal to RANDOM EFFECT then true                                                                                   |
| 38 | dependent                                           | string | dependent variable                                                                                                                      |
| 39 | adjusted_dependent                                  | string | reorganise dependent variable into smaller groups                                                                                       |
| 40 | independent                                         | string | independent variables                                                                                                                   |
| 41 | adjusted_independent                                | string | reorganise independent variable into smaller group                                                                                      |
| 42 | social                                              | string | if adjusted_independent in ENVIRONMENTAL AND SOCIAL, SOCIAL, CSP, CSR, ENVIRONMENTAL, SOCIAL and GOVERNANCE                             |
| 43 | environmnental                                      | string | if adjusted_independent in ENVIRONMENTAL, ENVIRONMENTAL AND SOCIAL, ENVIRONMENTAL, SOCIAL and GOVERNANCE                                |
| 44 | governance                                          | string | if adjusted_independent in GOVERNANCE ENVIRONMENTAL, SOCIAL and GOVERNANCE                                                              |
| 45 | lag                                                 | string | the table contains lag or not                                                                                                           |
| 46 | interaction_term                                    | string | the table contains interaction terms or not                                                                                             |
| 47 | quadratic_term                                      | string | the table contains quadratic terms or not                                                                                               |
| 48 | n                                                   | float  | number of observations                                                                                                                  |
| 49 | r2                                                  | float  | R square                                                                                                                                |
| 50 | beta                                                | float  | Beta coefficient                                                                                                                        |
| 51 | sign_of_effect                                      | string | if stars is not blank and beta > 0, then POSITIVE, if stars is not blank and beta < 0, then NEGATIVE else INSIGNIFICANT                 |
| 52 | sign_positive                                       | string | if sign_of_effect is POSITIVE then True                                                                                                 |
| 53 | sign_negative                                       | string | if sign_of_effect is NEGATIVE then True                                                                                                 |
| 54 | sign_insignificant                                  | string | if sign_of_effect is INSIGNIFICANT then True                                                                                            |
| 55 | significant                                         | string | is beta significant                                                                                                                     |
| 56 | final_standard_error                                | float  | reconstructed standard error and use sr when true_standard_error is nan or error                                                        |
| 57 | to_check_final                                      | string | Final check rows                                                                                                                        |

    