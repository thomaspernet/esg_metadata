
# esg metadata



The relationship between corporate social responsibility (CSR) and corporate financial performance (CFP) has been relentlessly investigated over the last fifty years with diverse conclusions. This paper develops the concept of CSR by meta-analyzing the empirical evidence available on the relationship between CSR and the financial performance of companies. Based on 1476 effects revealed from 111 studies published from 1997 to 2020, we find the probability of observing a positive direct effect to be higher when CSR is measured from the social viewpoint. Furthermore, examining seven categories of CSR while looking deeper into whether the focus is on philanthropy, ESG score, or specialized rating agencies, we find the effect of CSR to be statistically significant only for philanthropy and CSR disclosure categories. While the effect of CSR measured in philanthropy negatively affects CFP, the effect of disclosure on CFP is positive. Overall, our study suggests that the best performances are obtained when CSR is assessed through companies’ own disclosures, and also when firms focus on the social aspects of CSR rather than the environmental aspects. The results based on studies by regions indicate that the best performances are obtained for North American countries where there is a significant positive impact of CSR on CFP. This could suggest that the instrumental stakeholder theory could be supported through the visibility of CSR (from a social point of view) in North America. We have also noted that studies published after the 2009 crisis have a lower probability of finding positive results than those published before the crisis.

## Table of Content

 - **00_data_catalog/**
   - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog/README.md)
   - **HTML_ANALYSIS/**
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog/HTML_ANALYSIS/README.md)
     - [esg metadata - Tasks - Task title-- Analyse sign of effect and p value.html](https://htmlpreview.github.io/?https://github.com/thomaspernet/esg_metadata/blob/master/00_data_catalog/HTML_ANALYSIS/esg metadata - Tasks - Task title-- Analyse sign of effect and p value.html)
     - [papers_meta_analysis.html](https://htmlpreview.github.io/?https://github.com/thomaspernet/esg_metadata/blob/master/00_data_catalog/HTML_ANALYSIS/papers_meta_analysis.html)
   - **temporary_local_data/**
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/00_data_catalog/temporary_local_data/README.md)
 - **01_data_preprocessing/**
   - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/README.md)
   - **00_download_data/**
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/00_download_data/README.md)
     - **ESG/**
       - [esg_paper_information.py](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/00_download_data/ESG/esg_paper_information.py)
     - **METADATA_TABLES/**
       - [esg_metaanalysis.py](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/00_download_data/METADATA_TABLES/esg_metaanalysis.py)
     - **SCIMAGO/**
       - [scimagojr.py](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/00_download_data/SCIMAGO/scimagojr.py)
   - **01_transform_tables/**
     - [00_meta_analysis.md](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/01_transform_tables/00_meta_analysis.md)
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/01_transform_tables/README.md)
     - **Reports/**
       - [00_meta_analysis.html](https://htmlpreview.github.io/?https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/01_transform_tables/Reports/00_meta_analysis.html)
       - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/01_data_preprocessing/01_transform_tables/Reports/README.md)
 - **02_data_analysis/**
   - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/02_data_analysis/README.md)
   - **00_statistical_exploration/**
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/02_data_analysis/00_statistical_exploration/README.md)
   - **01_model_train_evaluate/**
     - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/02_data_analysis/01_model_train_evaluate/README.md)
     - **00_replicate_results/**
       - [00_replicate_tables.md](https://github.com/thomaspernet/esg_metadata/tree/master/02_data_analysis/01_model_train_evaluate/00_replicate_results/00_replicate_tables.md)
       - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/02_data_analysis/01_model_train_evaluate/00_replicate_results/README.md)
       - **Reports/**
         - [00_replicate_tables.html](https://htmlpreview.github.io/?https://github.com/thomaspernet/esg_metadata/blob/master/02_data_analysis/01_model_train_evaluate/00_replicate_results/Reports/00_replicate_tables.html)
 - **utils/**
   - [README.md](https://github.com/thomaspernet/esg_metadata/tree/master/utils/README.md)
   - [create_report.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/create_report.py)
   - [create_schema.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/create_schema.py)
   - [make_toc.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/make_toc.py)
   - [prepare_catalog.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/prepare_catalog.py)
   - [summary_stat.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/summary_stat.py)
   - [update_glue_github.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/update_glue_github.py)
   - **IMAGES/**
     - [script_diagram_meta_analysis_esg_cfp.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/IMAGES/script_diagram_meta_analysis_esg_cfp.py)
   - **latex/**
     - [latex_beautify.py](https://github.com/thomaspernet/esg_metadata/tree/master/utils/latex/latex_beautify.py)
