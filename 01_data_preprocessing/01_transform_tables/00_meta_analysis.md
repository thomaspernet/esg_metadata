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
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# US Name

Data preparation combine table information and journals


# Description

None

## Merge

**Main table** 

papers_meta_analysis_new

Merged with:

- papers_meta_analysis
- journals_scimago

# Target

- The file is saved in S3:
- bucket: datalake-datascience
- path: DATA/FINANCE/ESG/ESG_CFP
- Glue data catalog should be updated
- database: esg
- Table prefix: meta_analysis_
- table name: meta_analysis_esg_cfp
- Analytics
- HTML: ANALYTICS/HTML_OUTPUT/meta_analysis_esg_cfp
- Notebook: ANALYTICS/OUTPUT/meta_analysis_esg_cfp

# Metadata

- Key: 234_esg_metadata
- Epic: Dataset transformation
- US: Prepare meta-analysis table
- Task tag: #journal-information, #papers-information
- Analytics reports: https://htmlpreview.github.io/?https://github.com/thomaspernet/esg_metadata/blob/master/00_data_catalog/HTML_ANALYSIS/META_ANALYSIS_ESG_CFP.html

# Input

## Table/file

**Name**

- papers_meta_analysis_new
- papers_meta_analysis
- journals_scimago

**Github**

- https://github.com/thomaspernet/esg_metadata/blob/master/01_data_preprocessing/01_transform_tables/00_meta_analysis.md
<!-- #endregion -->
```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
from awsPy.aws_authorization import aws_connector
from awsPy.aws_s3 import service_s3
from awsPy.aws_glue import service_glue
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import os, shutil, json, re

from GoogleDrivePy.google_drive import connect_drive
from GoogleDrivePy.google_platform import connect_cloud_platform
from GoogleDrivePy.google_authorization import authorization_service

path = os.getcwd()
parent_path = str(Path(path).parent.parent)


name_credential = 'financial_dep_SO2_accessKeys.csv'
region = 'eu-west-2'
bucket = 'datalake-london'
path_cred = "{0}/creds/{1}".format(parent_path, name_credential)
```

```python inputHidden=false jupyter={"outputs_hidden": false} outputHidden=false
con = aws_connector.aws_instantiate(credential = path_cred,
                                       region = region)
client= con.client_boto()
s3 = service_s3.connect_S3(client = client,
                      bucket = bucket, verbose = True) 
glue = service_glue.connect_glue(client = client) 
```

```python
try:
    os.mkdir("creds")
except:
    pass

s3.download_file(key = "CREDS/Financial_dependency_pollution/creds/token.pickle", path_local = "creds")
s3.download_file(key = "CREDS/Financial_dependency_pollution/creds/service.json", path_local = "creds")

auth = authorization_service.get_authorization(
    path_credential_gcp=os.path.join(path, "creds", "service.json"),
    path_credential_drive=os.path.join(path, "creds"),
    verbose=False,
    scope=['https://www.googleapis.com/auth/spreadsheets.readonly',
           "https://www.googleapis.com/auth/drive"]
)
gd_auth = auth.authorization_drive(path_secret=os.path.join(
    path, "creds", "credentials.json"))
service_account = auth.authorization_gcp()
drive = connect_drive.drive_operations(gd_auth)


shutil.rmtree(os.path.join(path,"creds"))
```

```python
pandas_setting = True
if pandas_setting:
    cm = sns.light_palette("green", as_cmap=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
```

# Author information

During a presentation (Desir Seminar), it has been pointed out that characteristic of an author might impact the desire results. Most of the information are available from the internet. 

We use two sources of information:

- [Semantic scholar](https://www.semanticscholar.org/me/research)
    - API: https://www.semanticscholar.org/product/api

Using both data sourcs, we will retrieve or compute the following information:

- author information (name, affiliation, publication, email, etc) -> From the API 
- author gender: From a model
- author expertise in ESG: Computed 

The workflow works in three steps:

1. Train gender detection model using US name from the public dataset [USA Names](https://console.cloud.google.com/marketplace/product/social-security-administration/us-names?project=lofty-foundry-302615)
2. Download paper and author information from the spreadsheet [CSR Excel File Meta-Analysis - Version 4 - 01.02.2021](https://docs.google.com/spreadsheets/d/11A3l50yfiGxxRuyV-f3WV9Z-4DcsQLYW6XBl4a7U4bQ/edit?usp=sharing)
3. Fusion paper and author informations. The final table has the size of number of papers x number of authors per paper. 


## Train deep learning model gender detection

The first step of the workflow consists to train a basic LSTM architecture to deter the gender from family name. 

Training the model requires the following steps:

1. Download the data from Google Cloud Platform
2. Lowercase first name, split character and convert them to numerical value
3. Train the model using a vector embedding, Bidirectional LSTM layer and a dense layer to compute the prediction 

<!-- #region heading_collapsed="true" -->
### Download data

The data comes from the public dataset [USA Names](https://console.cloud.google.com/marketplace/product/social-security-administration/us-names?project=lofty-foundry-302615)
<!-- #endregion -->

```python
project = 'valid-pagoda-132423'
gcp = connect_cloud_platform.connect_console(project = project,
                                             service_account = service_account,
                                             colab = False)

sql = """
SELECT
  name,
  gender,
  COUNT(name) AS num_names
FROM
  `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY
  name,
  gender
"""

#names_df = client.query(sql).to_dataframe()
names_df = gcp.upload_data_from_bigquery(query = sql,location = "US")
```

```python
names_df.shape
```

```python
names_df.head()
```

<!-- #region heading_collapsed="true" -->
### Clean up and pre-process

1. Clean up name ->Lowercase the name.
2. Get a list of characters from the name.
3. Create a padding: each name as a length of 50 characters max. The padding fills the empty values with 0 to reach 50.
4. Encode characters and gender as specify in Keras. Note, the character 'space' is encoded as 0

For example, the preprocessing step does the following:

Take the name  "mary", the character "m" is given the number 13, the character "a" is 1, and so one. The 0s are the padding because the matrix should have the same dimension

![image.png](attachment:477f5f29-e1e5-4a28-85fc-e6effb0eb8b5.png)
<!-- #endregion -->

```python
def preprocess(names_df,column, train=True, to_lower = True):
    # Step 1: Lowercase
    if to_lower:
        names_df['name'] = names_df[column].str.lower()
    else:
        names_df['name'] = names_df[column]
    # Step 2: Split individual characters
    names_df['name'] = [list(name) for name in names_df['name']]

    # Step 3: Pad names with spaces to make all names same length
    name_length = 50
    names_df['name'] = [
        (name + [' ']*name_length)[:name_length] 
        for name in names_df['name']
    ]

    # Step 4: Encode Characters to Numbers
    names_df['name'] = [
        [
            max(0.0, ord(char)-96.0) 
            for char in name
        ]
        for name in names_df['name']
    ]
    
    if train:
        # Step 5: Encode Gender to Numbers
        names_df['gender'] = [
            0.0 if gender=='F' else 1.0 
            for gender in names_df['gender']
        ]
        return names_df
    else:
        return names_df['name']
```

```python
names_df = preprocess(names_df, column = 'name', train=True)
names_df.head()
```

<!-- #region heading_collapsed="true" -->
### Model Architecture

1. Embedding layer: to “embed” each input character’s encoded number into a dense 256 dimension vector..
2. Bidirectional LSTM layer: .
3. Final Dense layer: Prediction 0/1 for male/female

Note: Embedding layer enables us to convert each word into a fixed length vector of defined size. The resultant vector is a dense one with having real values instead of just 0’s and 1’s. The fixed length of word vectors helps us to represent words in a better way along with reduced dimensions
<!-- #endregion -->

```python
#!pip install --upgrade tensorflow
```

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def lstm_model(num_alphabets=27, name_length=50, embedding_dim=256):
    model = Sequential([
        Embedding(num_alphabets, embedding_dim, input_length=name_length),
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model
```

The idea is to turns positive integers (indexes) into dense vectors of fixed size. Then this layer can be used as the first layer in a model.

The size of the vocabulary (the list from the preprocessing) is equal to 27: The alphabet has 26 letters, and the space characters. We want the output layer to be a vector of 256 weights. `input_length` is the maximum size of the name. We can set it up since the length of input sequences is constant.

Here is an example of how the vector embedding output looks like

Note: In this example we have not trained the embedding layer. The weights assigned to the word vectors are initialized randomly.

```python
model_ex = Sequential()
model_ex.add(Embedding(input_dim= 27, output_dim= 256, input_length=50))
model_ex.compile(loss = 'binary_crossentropy', metrics= 'accuracy')
output_array = model_ex.predict(names_df['name'].values.tolist())
```

```python
output_array.shape
```

The embedding vector for the first word is:

```python
output_array[0].shape
```

```python
output_array[0]
```

<!-- #region heading_collapsed="true" -->
### Training the Model

We’ll use the standard tensorflow.keras training pipeline
<!-- #endregion -->

```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
```

```python
%time

# Step 1: Instantiate the model
model = lstm_model(num_alphabets=27, name_length=50, embedding_dim=256)

# Step 2: Split Training and Test Data
X = np.asarray(names_df['name'].values.tolist())
y = np.asarray(names_df['gender'].values.tolist())

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

# Step 3: Train the model
callbacks = [
    EarlyStopping(monitor='val_accuracy',
                  min_delta=1e-3,
                  patience=5,
                  mode='max',
                  restore_best_weights=True,
                  verbose=1),
]

history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

# Step 4: Save the model
model.save('MODELS_AND_DATA/boyorgirl.h5')
```

```python
# Step 5: Plot accuracies
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
```

# Semantic scolar

Our primary objective is to get the information about gender, evaluate the expertise of an author about ESG but also derive meaningful information from the abstract. The data source [Semantic Scholar](https://www.semanticscholar.org/me/research) has 198,182,311 papers from all fields of science.

Our strategy is to use the API to search for a paper in order to get the related information (DOI, cite, performance) and more importantly, the ID of the author(s). Indeed, to get information about an author, we need to know his/her ID. As soon as we have the ID, we can collect and compute all other information (i.e. gender and expertise)

The workflow is the following:

1. Download data and predict gender
2. Flag ESG papers
3. Compute sentiment and cluster papers using the abstract
4. Compute ESG expertise score
5. Combine all information

![](https://cdn-images-1.medium.com/max/1600/1*rEcC_x1CRlWx1KRCySotIg.png)

```python
#from serpapi import GoogleSearch
from tqdm import tqdm
import time
import pickle
import re
from tensorflow.keras.models import load_model
import unicodedata
import requests
```

## 1. Download data and predict gender

We follow four steps approaches to get the paper and author information 

1. We fetch the data from the spreadsheet [CSR Excel File Meta-Analysis — Version 4–01.02.2021](https://docs.google.com/spreadsheets/d/11A3l50yfiGxxRuyV-f3WV9Z-4DcsQLYW6XBl4a7U4bQ/edit?usp=sharing) (note, I use the library [GoogleDrive-python](https://github.com/thomaspernet/GoogleDrive-python) to get the data from the spreadsheet). 
2. We pass the paper’s title into Semantic Scholar API to find the paper’s ID and use the ID to download the paper information (including the authors’ ID)
3. We pass the author’s ID into Semantic Scholar API to download the author’s information
4. We predict the gender from the first name of the author

![](https://cdn-images-1.medium.com/max/1600/1*jsBIqWumazLCekrlKbUg_A.png)


### Step 1: Download data from Google spreadsheet

The original data was collected on a Google spreadsheet ([CSR Excel File Meta-Analysis — Version 4–01.02.2021](https://docs.google.com/spreadsheets/d/11A3l50yfiGxxRuyV-f3WV9Z-4DcsQLYW6XBl4a7U4bQ/edit?usp=sharing)) with some relevant information but it also contains errors. The detection of errors has been done separately and won’t be covered in this post. From this spreadsheet, we will only use the title and the publication name. All the other information is discarded and will be retrieved using Semantic Scholar. 

```python
pred_model = load_model('MODELS_AND_DATA/boyorgirl.h5')
```

```python
#!pip install google-search-results
```

```python
FILENAME_SPREADSHEET = "CSR Excel File Meta-Analysis - Version 4 -  01.02.2021"
spreadsheet_id = drive.find_file_id(FILENAME_SPREADSHEET, to_print=False)
doi = drive.download_data_from_spreadsheet(
    sheetID = spreadsheet_id,
    sheetName = "Feuil1",
    to_dataframe = True)
```

```python
doi.loc[lambda x: x['Title'].isin(['The corporate social performance–financial performance link'])].head(1)
```

### Steps 2–3: paper information and author ID

In the second step, we want to use the unique title’s name from the spreadsheet to get the information we need (gender, abstract, publication year, etc). 

The previous image show one paper written by S. Waddock and S. Graves. We can look at the paper in [Semantic Scholar](https://www.semanticscholar.org/me/research)

- [The corporate social performance-financial performance](https://www.semanticscholar.org/paper/2e899bc9e49e4a55374f26fdfd3f777658d460ab)

![](https://cdn-images-1.medium.com/max/1600/1*RsOxfDRLtIu-_pUbmn--pw.png)

The DOI is “*10.1002/(SICI)1097–0266(199704)18:4<03::AID-SMJ869>3.0.CO;2-G*”

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
}

```

```python
field = [
    "url",
    "title",
    "abstract",
    "venue",
    "year",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "fieldsOfStudy",
    "authors"]
field_paper = [
    "externalIds",
    "url",
    "title",
    "abstract",
    "venue",
    "year",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "fieldsOfStudy",
    "authors"
]
field_authors = [
    "externalIds",
    "url",
    "name",
    "aliases",
    "affiliations",
    "homepage",
    "papers"
]
```

```python
def find_doi(paper_name):
    """
    to keep thing simple, assume first results in the best option
    """
    paper_name_clean = (
        paper_name
        .lower()
        .replace("  ", "+")
        .replace(" ", "+")
        .replace("\n", "+")
        .replace(",", "+")
        .replace("–", "")
        .replace("++", "+")
        .replace(":", "")
    )
    url_paper = 'https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields={}'.format(
        paper_name_clean, ",".join(field))
    response_1 = requests.get(url_paper, headers=headers)
    if response_1.status_code == 200:
        response_1 = response_1.json()
        if len(response_1['data']) > 0:
            url_paper = "https://api.semanticscholar.org/graph/v1/paper/{}?fields={}".format(
                response_1['data'][0]['paperId'], ",".join(field_paper))
            response_2 = requests.get(url_paper, headers=headers)
            if response_2.status_code == 200:
                # find publication name because not available in the API
                publication_name = (
                    doi
                    .loc[lambda x: x['Title'].isin([paper_name])]
                    .reindex(columns=['Publication name'])
                    .drop_duplicates()
                    .values[0][0]
                )
                response_2 = response_2.json()
                response_2['paper_name_source'] = paper_name
                response_2['publication_name'] = publication_name
                response_2['status'] = 'found'

                # find authors details information
                authors_fulls = []
                for aut in response_1['data'][0]['authors']:
                    url_author = 'https://api.semanticscholar.org/graph/v1/author/{}?fields={}'.format(
                        aut['authorId'],
                        ",".join(field_authors))
                    response_3 = requests.get(url_author, headers=headers)
                    if response_3.status_code == 200:
                        authors_fulls.append(response_3.json())

                if len(authors_fulls) > 0:
                    response_2['authors_detail'] = authors_fulls

                return response_2
            else:
                return {'paper_name': paper_name, 'status': 'not found'}
        else:
            return {'paper_name': paper_name, 'status': 'not found'}
    else:
        return {'paper_name': paper_name, 'status': 'not found', 'status_code': response_1.status_code}
```

```python
def clean_name(name='Sarah'):
    """
    """
    
    return "".join(
        (
            c
            for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )
    ).lower().replace("-", ' ')
def prediction_gender(name=["sarah"]):
    """
    name should be normalised and a list of candidates
    """
    return np.mean(pred_model.predict(
        np.asarray(
            preprocess(
                pd.DataFrame(
                    name,
                    columns=['semantic_0']
                ), column="semantic_0", train=False
            )
            .values.tolist()
        )
    ))
```

To find the information from steps 2 to 3, we need to use 2 different [APIs](https://api.semanticscholar.org/graph/v1):

**Publication**

- To find the paper ID, we use https://api.semanticscholar.org/graph/v1/paper/search?query=

we construct the URL by cleaning the title, and we explicitly add the list of the fields to be returned:

https://api.semanticscholar.org/graph/v1/paper/search?query=the+corporate+social+performancefinancial+performance+link&fields=url,title,abstract,venue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,fieldsOfStudy,authors

You can copy/paste the URL in your web browser to see all the information.

The response gives the paper ID:

```
{"paperId": "2e899bc9e49e4a55374f26fdfd3f777658d460ab", "url": "https://www.semanticscholar.org/paper/2e899bc9e49e4a55374f26fdfd3f777658d460ab", "title": "The corporate social performance-financial performance link"
```

but also, the authors( and ID)

```
"authors": [{"authorId": "66042905", "name": "S. Waddock"}, {"authorId": "2367938", "name": "S. Graves"}
```

Note that, we need an intermediary API call to get the DOI from the following URL https://api.semanticscholar.org/graph/v1/paper/2e899bc9e49e4a55374f26fdfd3f777658d460ab?fields=externalIds,url,title,abstract,venue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,fieldsOfStudy,authors

**Authors**

- We use the following API to download the information of each author https://api.semanticscholar.org/graph/v1/author/66042905?fields=externalIds,url,name,aliases,affiliations,homepage,papers

The response provides two information we will use to compute the gender and the ESG score:

- *aliases*: all possible names of the author's
- *papers*: All papers from the authors. From the image below, we can see the author S. Waddock has 313 publications

![](https://cdn-images-1.medium.com/max/1600/1*caiIQivJ1fR2AJIPWxR_Jg.png)

https://www.semanticscholar.org/author/S.-Waddock/66042905?sort=influence

in the response, we store all of the 313 publications because we will use this information to compute the ESG expertise score.

```
[{'authorId': '66042905',
   'externalIds': {},
   'url': 'https://www.semanticscholar.org/author/66042905',
   'name': 'S. Waddock',
   'aliases': ['S Waddock',
    'Sandra Waddock',
    'Sandr A Waddock',
    'Sandra A. Waddock'],
   'affiliations': [],
   'homepage': None,
   'papers': [{'paperId': '39657170f4d0496f79d7c766e1911c48e5b8f25c',
     'title': 'The UN Guiding Principles on Business and Human Rights: Implications for Corporate Social Responsibility Research'}, ....]
```

```python
response = find_doi(paper_name = list(doi['Title'].unique())[-2]) 
response['externalIds']['DOI']
```

### Steps 4: Gender prediction

In the next step, we want to predict the gender of the author. The first author is *S. Waddock* which is impossible to detect the gender because only one letter displays for the first name. Therefore, we will combine the first name with all the aliases. We add another constraint, the first name should have more than 2 characters:

- ‘S. Waddock’: discarded
- ‘S Waddock’: discarded
- ‘Sandra Waddock’,
- ‘Sandr A Waddock’,
- ‘Sandra A. Waddock’

Then we push all the candidates to the model and return the average probability. The model gives an average probability of 43%, meaning the author is a female.

```python
prediction_gender(
    list(
                dict.fromkeys(
                    [clean_name(name=
                                re.sub(r"[^a-zA-Z0-9]+", ' ', a
                                      ).split(" ")[0]) for a in 
                     [response['authors_detail'][0]['name']] + response['authors_detail'][0]['aliases']
                     if len(a.split(" ")[0]) >2]
                )
            )
)
```

Get full list of information

```python
list_paper_semantic = []
list_failure = []
```

```python
for i, p in tqdm(enumerate(list(doi['Title'].unique()))):
    time.sleep(15)
    response = find_doi(paper_name = p)  
    if response['status'] == 'found':
        list_paper_semantic.append(response)
    else:
        list_failure.append(p)
```

```python
for i, authors in tqdm(enumerate(list_paper_semantic)):
    for author in authors["authors_detail"]:
        #### Clean authors
        author_clean = [clean_name(name=author["name"].split(" ")[0])] if \
                        len(clean_name(name=author["name"].split(" ")[0]))>2 else None
        if author["aliases"] is not None:
            author_clean_alias = list(
                dict.fromkeys(
                    [clean_name(name=
                                re.sub(r"[^a-zA-Z0-9]+", ' ', a
                                      ).split(" ")[0]) for a in author["aliases"] if len(a.split(" ")[0]) >2]
                )
            )
            if author_clean is not None:
                author_clean.extend(author_clean_alias)
            else:
                author_clean = author_clean_alias
        #### predict gender
        if len(author_clean) > 0:
            max_prediction = prediction_gender(name=author_clean)
            gender= "MALE" if max_prediction >=.5 else "FEMALE"
        else:
            max_prediction = None
            gender = 'UNKNOWN'
        author['gender'] = {'gender': gender, 'probability':max_prediction}
```

```python
list_failure
```

Failure: 

- 'The Effect of Corporate Social Responsibility on Financial Performance: Evidence from the Banking Industry in Emerging Economies',
- 'An examination of corporate social responsibility and financial performance: A study of the top 50 Indonesian listed corporations',
- 'Does it pay to be different? An analysis of the relationship between corporate social and financial performance (',
- 'Corporate Social and Environmental Performance and Their Relation to Financial Performance and Institutional Ownership: Empirical Evidence on Canadian Firms',
-  'The Corporate Social-Financial Performance Relationship: A Typology and Analysis'

```python
list_failure = ['The Effect of Corporate Social Responsibility on Financial Performance: Evidence from the Banking Industry in Emerging Economies',
 'An examination of corporate social responsibility and financial performance: A study of the top 50 Indonesian listed corporations',
 'Does it pay to be different? An analysis of the relationship between corporate social and financial performance (',
 'Corporate Social and Environmental Performance and Their Relation to Financial Performance and Institutional Ownership: Empirical Evidence on Canadian Firms',
 'The Corporate Social-Financial Performance Relationship: A Typology and Analysis']
```

```python
for ind, paper in enumerate(list_paper_semantic):
    with open("paper_id_{}".format(paper["paperId"]), "w") as outfile:
        json.dump(eval(str(paper)), outfile)
    s3.upload_file(
        file_to_upload="paper_id_{}".format(paper["paperId"]),
        destination_in_s3="DATA/JOURNALS/SEMANTIC_SCHOLAR/PAPERS",
    )
    os.remove("paper_id_{}".format(paper["paperId"]))
```

```python
# Store data (serialize)
with open('MODELS_AND_DATA/list_paper_semantic.pickle', 'wb') as handle:
    pickle.dump(list_paper_semantic, handle)
```

## 2. Flag ESG paper

The list of papers we saved in the previous steps contains 266 authors, with 14,443 unique publications. For each author, we want to evaluate how familiar he/she is with the topic of ESG. To flag an ESG publication (14.443), we rely on a naive technique.

The technique is the following:

- Create a clean list of words from the title (removing English stop words, special characters, and lower case)
- Flag if the clean list contains “esg”, ”environmental”, ”social”, ”governance”

The image below shows how we use the technique to flag the ESG paper. Take the title “*Corporate social responsibility and firm value: Guiding through
economic policy uncertainty*”, after the cleaning process, we end up with the following list of keywords “[corporate, social, responsibility, firm, value, guiding, economic, policy, uncertainty]”. Since the list contains the word “social”, we flag it as an ESG topic. By analogy, the title “*Does financial development really spur nascent entrepreneurship in Europe? A panel data analysis*” does not contain any of the ESG keywords

In total, 2094 papers deal with ESG among the 14,443 papers (14%). We will use this information later to construct the ESG expertise score.

```python
list_paper_semantic = pickle.load( open( "MODELS_AND_DATA/list_paper_semantic.pickle", "rb" ))
```

```python
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from scipy.spatial import distance
import string
#nltk.download('stopwords')
stop = stopwords.words('english')
```

```python
def basic_clean(text):
    return re.sub(r'[^\w\s]|[|]', '', text).split()

def dumb_search(items):
    #return any(item in 'esg environmental social governance' for item in items)
    return True if len([i for i in ['esg',"environmental","social","governance"] if i in items]) > 0 else False
```

```python
list_papers = []
null = [list_papers.extend(i) for i in pd.json_normalize(list_paper_semantic, "authors_detail")['papers'].to_list()]
all_connected_paper = (
    pd.DataFrame(list_papers)
    .drop_duplicates()
    .assign(
        name_clean=lambda x:
        x.apply(lambda x:
                basic_clean(' '.join([word.lower() for word in x['title'].split() if word not in (stop)])), axis=1)
    )
    .assign(
        esg = lambda x: x.apply(
            lambda x:
            dumb_search(x['name_clean']), axis = 1)
    )
)
all_connected_paper.shape
```

```python
all_connected_paper.head()
```

```python
all_connected_paper['esg'].value_counts(normalize = True)
```

```python
all_connected_paper['esg'].value_counts(normalize = False)
```

## 3. Sentiment and cluster papers

The last batch of information relates to the pertinence and details of the abstract. We might think that the abstract contains information about the “quality” or “emotion” behind the paper. Therefore, we propose to compute the following variables:

- **sentiment**: positive or negative. The overall feeling of the abstract. Positive means the abstract tend to have more words associated with a positive connotation.
- **cluster**: 3 clusters computed using the words in the abstract (embeddings), the number of verbs, nouns, and adjectives but also the size of the abstract.

### Construct sentiment

We use the brilliant [Flair](https://github.com/flairNLP/flair) library to compute the sentiment for different reasons. First of all, we don’t have labels in the abstract, hence we cannot train our own model. Second, Flair uses state-of-the-art NLP architecture to train their model, meaning that it gives far better results than if we had to build our model. 

The workflow to get the sentiment is the following

- Step 1: Clean the abstract:
   — Lowercase words
   — Remove [+XYZ chars] in content
   — Remove multiple spaces in content
   — Remove ellipsis (and last word)
   — Replace dash between words
   — Remove punctuation
   — Remove stopwords
   — Remove digits
   — Remove short tokens
- Step 2: Compute the sentiments using the [Flair](https://github.com/flairNLP/flair) library

Among the 106 papers we have, 71 have a positive sentiment derived from the abstract and 35 negative ones.

### Clustering

The abstract contains relevant information about the quality of the papers, and we want to extract them to group the papers into 3 clusters. To construct the cluster, we use the vector word embedding from “*word2vec-google-news-300”*. We don’t have enough data to train our model, and Google has already done the heavy lifting so it sounds more reasonable to use pre-computed vectors. We also include the number of ESG occurrences in the abstract, number of adjectives, nouns, and verbs. It seems plausible that the “quality” of the abstract is correlated with the number of verbs or adjectives since they give more emotion to the reader.

- Step 1: Count the number of adjectives, nouns, and verbs
- Step 2: Get the vector’s embedding from the pre-trained model `word2vec-google-news-300` and look up each word in the list. Compute the average to get a vector of 100 weights for a given document
- Step 3: Standardize the number of occurrences, verbs, nouns, and adjectives. Beforehand, we normalize the number of each occurrence over the length of the abstract
- Step 4: Compute the cluster using K-mean


```python
#!pip install flair
```

```python
from flair.models import TextClassifier
from flair.data import Sentence
import gensim.downloader as api
import nltk

wv = api.load('word2vec-google-news-300')
```

```python
def clean_text(text, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.
        

    Returns:
        Tokenized text.
    """
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation
    text = text.replace('abstract', '')
    text = text.split()

    #tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in text if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens
```

```python
def average_embedding(doc, stopwords, list_embedding):
    text = clean_text(doc, stopwords)
    #### lenght
    lenght = len([i for i in ['esg', "environmental", "social", "governance"] if i in
                  sorted(text)])
    
    ### sentiment
    sentiment = TextClassifier.load('en-sentiment')
    sentence = Sentence(" ".join(text))
    sentiment.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        sent = "POSITIVE"
    elif "NEGATIVE" in str(score):
         sent =  "NEGATIVE"
    else:
         sent =  "NEUTRAL"
            
    #### 
    test = nltk.pos_tag(text)
    list_tags = {
        'ADJ':0,
        'NOUN':0,
        'VERB':0
    }
    for g in test:
        if g[1] in ["JJ", "JJR", "JJS"]:
            list_tags['ADJ']+=1
        elif g[1] in ["NN", "NNS", "NNP" ,"NNPS" ]:
            list_tags['NOUN']+=1
        elif g[1] in ["VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]:
            list_tags['VERB']+=1

    text = list(dict.fromkeys(text))
    weights = np.mean([list_embedding[i] for i in text if i in list_embedding],axis=0)
    return {
       'weights':weights,
        'lenght':lenght,
        'size': len(doc.split()) if doc is not None else None,
        'sentiment':sent,
        'tag':list_tags
    }
```

### Sentiment

```python
%%time
df_embedding = (
    pd.DataFrame(list_paper_semantic)
    .reindex(columns=['paperId', 'abstract'])
    .assign(
        avg_embedding=lambda x: x.apply(
            lambda x:
            average_embedding(
                doc=x['abstract'],
                stopwords=stop,
                list_embedding=wv)
            , axis=1
        )
    )
    .assign(
        weights = lambda x: x.apply(
            lambda x: x['avg_embedding']['weights'],
            axis = 1
        ),
        lenght = lambda x: x.apply(
            lambda x: x['avg_embedding']['lenght'],
            axis = 1
        ),
        sentiment = lambda x: x.apply(
            lambda x: x['avg_embedding']['sentiment'],
            axis = 1
        ),
        size = lambda x: x.apply(
            lambda x: x['avg_embedding']['size'],
            axis = 1
        ),
        adj = lambda x: x.apply(
            lambda x: x['avg_embedding']['tag']['ADJ'],
            axis = 1
        ),
         noun = lambda x: x.apply(
            lambda x: x['avg_embedding']['tag']['NOUN'],
            axis = 1
        ),
         verb = lambda x: x.apply(
            lambda x: x['avg_embedding']['tag']['VERB'],
            axis = 1
        )
    )
)
```

```python
df_embedding['sentiment'].value_counts()
```

### Clustering

```python
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import preprocessing
from sklearn import manifold
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

```python
def standardized(x, option = 1):
    if option == 1:
        mean = np.mean(x)
        sd = np.std(x)
        s = (x - mean)/sd
    else:
        s = (x - x.min())/(x.max() - x.min())
    return s
```

```python
scaler = StandardScaler()
le = preprocessing.LabelEncoder()
sc = preprocessing.StandardScaler()
df_tsne = (
    df_embedding.set_index("paperId").drop(
        columns=["abstract", "avg_embedding", "sentiment", "lenght", "adj","noun","verb", "size"]
    )
    .explode('weights')
    .reset_index()
    .assign(
        temps = 1,
        id_ = lambda x: x.groupby(['paperId'])['temps'].transform('cumsum')
    )
    .set_index(['paperId', 'id_'])
    .drop(columns = ['temps'])
    .unstack(-1)
    .droplevel(level=0, axis=1)
    .apply(pd.to_numeric)
    .reset_index()
    .set_index('paperId')
    .merge(
        (
            df_embedding
            .set_index("paperId")
            .reindex(columns=['sentiment', "lenght", "abstract", "adj","noun","verb", 'size'])
        ),
        left_index=True, right_index=True
    )
    .assign(
        sentiment = lambda x: le.fit_transform(x["sentiment"]),
    )
)
df_tsne = (
    df_tsne
    .assign(
        **{
            "{}".format(col): standardized(df_tsne[col]/df_tsne['size'], option = 2)
            for col in ['lenght','adj','noun', 'verb']
        },
        size = lambda x: standardized(x['size'])
    )
    .loc[lambda x: ~x['abstract'].isin([None])]
    .loc[lambda x: ~x.index.isin([
        '0bf6400dcc8d2a9c1b02c650cc8e0ebfedf99670',
        '732b67567b0ab51ca047fa0f3ebc89de29bbc8a4',
        'd9041bee67c6cfacc2e28b66e5702d7141648816'
    ])]
)
```

The input fed into k-mean looks like the image below. There is a vector of 300 values corresponding to the word embedding, and four other features capturing the occurrences.

```python
df_tsne.drop(columns = ["abstract"]).head().iloc[:3, -10:].drop(columns = ['sentiment',  'size' ])
```

In the end, we have three clusters, with 32 observations in cluster 0, 29 in cluster 1, and 38 in cluster 2. 

```python
kmeans_w_emb = KMeans(n_clusters=3, random_state=1).fit(
    df_tsne.drop(columns = ["abstract", 'sentiment', 
                            'size'
                           ])
)
pd.Series(kmeans_w_emb.labels_).value_counts()
```

```python
df_tsne = (
    df_tsne
    .drop(columns=['sentiment', "lenght", "abstract", "adj","noun","verb", 'size'])
    .merge(
        (
            df_embedding
            .set_index("paperId")
            .reindex(columns=['sentiment', "lenght", "abstract", "adj","noun","verb", 'size'])
        ),
        left_index=True, right_index=True
    )
    .reindex(columns=["cluster_w_emb", 'sentiment', "lenght", "abstract", "adj","noun","verb", 'size'])
    .assign(
        pct_adj = lambda x: x['adj']/ x['size'],
        pct_noun = lambda x: x['noun']/ x['size'],
        pct_verb = lambda x: x['verb']/ x['size'],
        cluster_w_emb = kmeans_w_emb.labels_
    )
)
```

Cluster 0 and 2 has more or less the same percentage of positive sentiments but cluster 1 leans toward positive sentiments.

```python
pd.concat(
    [
        (
            df_tsne.groupby("cluster_w_emb")["sentiment"]
            .value_counts()
            .unstack(-1)
            .assign(total=lambda x: x.sum(axis=1))
        ),
        (
            df_tsne
            .groupby("cluster_w_emb")["sentiment"]
            .value_counts()
            .unstack(-1)
            .assign(total=lambda x: x.sum(axis=1))
            .apply(lambda x: x / x["total"], axis=1)
            .drop(columns=["total"])
        ),
    ],
    axis=1,
)
```

Cluster 1 has on average abstracts longer than the two other clusters (165 words vs 135/147) but contains much fewer occurrences of ESG. Cluster 2 is the most descriptive with on average 18 verbs.

```python
for v in ["lenght","verb","adj","noun", "size"]:
    display(
    df_tsne
    .groupby('cluster_w_emb')
    .agg(
        {
            v:'describe'
        }
    )
)

```

If we compare two abstracts drawn from clusters 0 and 2 we can catch the differences. First of all, the first paper (cluster 2), has much more words (193 vs. 128) and contains twice as many verbs as the second paper. If we read the abstract in detail, we can measure the sensibility of the first paper. It provides more details and is more convincing than the second one, indicating potentially a better "quality".

```python
pd.concat(
    [
        (
            df_tsne.loc[
                lambda x: x.index.isin(["02281aebff7110c8b6efb59ebba448ecb7e2a4cc"])
            ]
            .head(1)
            .reindex(
                columns=[
                    "cluster_w_emb",
                    "abstract",
                    "lenght",
                    "verb",
                    "adj",
                    "noun",
                    "size",
                ]
            )
        ),
        (
            df_tsne.loc[
                lambda x: x.index.isin(["128fd0154eeaf6189fcff693abbd076aad42b900"])
            ]
            .head(1)
            .reindex(
                columns=[
                    "cluster_w_emb",
                    "abstract",
                    "lenght",
                    "verb",
                    "adj",
                    "noun",
                    "size",
                ]
            )
        ),
    ],
    axis=0,
)
```

## 4. CNRS journal ranking

Our intuition is that the journal ranking matters to finding a statistical link (or not) between ESG and CFP. If this is the case, there is a publication bias in the data. To validate our assumption, we rely on two different metrics:

1. **SJR**: The SCImago Journal Rank indicator is a measure of the scientific influence of scholarly journals that accounts for both the number of citations received by a journal and the importance or prestige of the journals where the citations come from
2. **CNRS journal ranking**: The **French National Centre for Scientific Research** (French: *Centre national de la recherche scientifique*, **CNRS**) is the French state research organization and is the largest fundamental science agency in Europe. Each year, the CNRS releases a ranking for more than 1256 journals. The CNRS ranks the journal into 4 categories, ranging from 1 as the best journals and 4 as the lowest. 

We have already downloaded the data from the [Scimago database](https://www.scimagojr.com/) and stored it in AWS S3. However, the CNRS does not have an available dataset (at least in CSV or spreadsheet format), therefore we need to use the release publication from the official website to get the ranking. 

The official releases are available at this [URL](https://www.gate.cnrs.fr/spip.php?rubrique31&lang=en). For our research, we will use the most recent release, dating from 2020: [*CNRS Journal Ranking in Economics and Management June 2020*](https://www.gate.cnrs.fr/IMG/pdf/categorisation37_liste_juin_2020-2.pdf)*.* The PDF contains more than 80 tables, with the publication’s name, ISSN, domain, and rank.

![](https://cdn-images-1.medium.com/max/1600/1*qT9-_dcn7AlTC94WRpH_-w.png)

```python
### release memory from large google word file
del wv
```

```python
# use terminal if cannot allocate memory
#!sudo yum install java-1.8.0-openjdk -y
#!pip install tabula-py
```

```python
import tabula
import requests
```

```python
url = "https://www.gate.cnrs.fr/IMG/pdf/categorisation37_liste_juin_2020-2.pdf"
r = requests.get(url, allow_redirects=True)
open('categorisation37_liste_juin_2020-2.pdf', 'wb').write(r.content)
```

To extract the information, we rely on the [tabula library](https://pypi.org/project/tabula-py/) which is a simple Python wrapper of [tabula-java](https://github.com/tabulapdf/tabula-java), and can read tables in a PDF. 

After converting the PDF into a Pandas dataframe, we get an extensive list of journals-ranking (see image below)

```python
list_tables = tabula.read_pdf('categorisation37_liste_juin_2020-2.pdf', pages='all')
list_list_tables= [list_tables[i].values.tolist() for i in range(0, len(list_tables)) if len(list_tables[i])>0]
df_cnrs = (
    pd.DataFrame([item for sublist in list_list_tables for item in sublist], columns = [
    "NAME","ISSN","DOMAINE","RANK"
])
    .assign(
        publication_name = lambda x:x['NAME'].str.lower()
    )
    .drop_duplicates()
)
df_cnrs.head(10)
```

```python
df_cnrs.shape
```

```python
df_publication_rank = (
    df_cnrs.merge(
        (
            pd.DataFrame(list_paper_semantic)
            .reindex(columns=["publication_name"])
            .drop_duplicates()
            .rename(columns={"Publication name": "publication_name"})
            .assign(publication_name=lambda x: x["publication_name"].str.lower())
            .drop_duplicates()
            .replace(
                {
                    "publication_name": {
                        "brq business research quarterly": "business research quarterly"
                    }
                }
            )
            .replace("\&", "and", regex=True)
        ),
        how="right",
        indicator=True,
    )
    .sort_values(by=["_merge"])
    .assign(
        RANK=lambda x: x["RANK"].astype("str"),
        rang_digit=lambda x: x["RANK"].str.extract("(\d+)"),
    )
    .assign(rang_digit=lambda x: x["rang_digit"].fillna("5"))
    .reindex(columns=["publication_name", "rang_digit"])
    .drop_duplicates()
    .sort_values(by=["rang_digit"], ascending=True)
)
df_publication_rank.head()
```

The list of journals we collected is available from *steps 2–3: paper information and author ID*. The comparison between the two files is trivial and we end up with the following CNRS ranking distribution:

- 37 journals are missing from the CNRS ranking
- 10 belongs to the second tiers
- 9 to the top tiers
- 3 in the fourth tiers
- 2 in the third tiers

I admit the paper collection process wasn’t as scientific as one can expect. I also cannot deny that the data might have a selection bias too, but it is something that was beyond my reach when I joined the project.

```python
df_publication_rank['rang_digit'].value_counts()
```

## 5. Wrapping up 

We managed to construct the following information from the list of papers available in Google Spreadsheet ([CSR Excel File Meta-Analysis — Version 4–01.02.2021](https://docs.google.com/spreadsheets/d/11A3l50yfiGxxRuyV-f3WV9Z-4DcsQLYW6XBl4a7U4bQ/edit?usp=sharing)):

- Papers information
- Authors information
- Sentiment derived from the abstract
- Clustering derived from the abstract
- Journal ranking from the CNRS

The final step of the strategy consists to merge every piece of information together into a single dataframe. The task is straightforward since we already have all the authors for a given paper. The “[*explode*](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html)” function from Pandas takes every author from the dictionary and creates a row for each author (if the paper has two authors, then the function creates two rows, one for the first author, and another one for the second). The other information can be merged using the paper ID or the publication name. 

At last, we can compute the ESG expertise score for each author by dividing the number of papers dealing with ESG over the total number of papers published. 

Our dataset is composed of 182 males and 83 females. The average author ESG expertise score is 0.22, with 75% of the data falling under 30%.

```python
def count_esg(papers):
    """
    papers list with keys paperId and title
    """
    return sum([all_connected_paper.loc[lambda x: 
                                        x['paperId'].isin([item['paperId']])]['esg'].values[0] for item in papers])
```

```python
df_temp =(
            pd.json_normalize(list_paper_semantic, "authors_detail")
            .assign(
                name=lambda x: x["name"].str.lower(),
                semantic=lambda x: x.apply(
                    lambda x: "".join(
                        (
                            c
                            for c in unicodedata.normalize("NFD", x["name"])
                            if unicodedata.category(c) != "Mn"
                        )
                    ),
                    axis=1,
                ),
            )
            .drop_duplicates(subset=["name"])
            .assign(
                total_paper=lambda x: x["papers"].str.len(),
                esg=lambda x: x.apply(
                    lambda x: count_esg(x["papers"]), axis=1),
                pct_esg=lambda x: x["esg"] / x["total_paper"],
            )
        )
```

```python
df_temp['pct_esg'].describe()
```

```python
df_temp['pct_esg'].loc[lambda x:x> 0].plot.hist(bins=12, alpha=0.5, title = 'Distribution author ESG expertise',figsize=(10,8))
```

```python
df_temp['gender.gender'].value_counts()
```

```python
df_authors_journal_full = (
    # Journal list
    pd.json_normalize(list_paper_semantic, meta=["externalIds"])
    .rename(columns={"authors_detail": "author_details_semantic"})
    .drop(columns=["paper_name_source"])
    .assign(
        nb_authors=lambda x: x["authors"].str.len(),
        authors_list=lambda x: x.apply(
            lambda x: [i["name"] for i in x["authors"] if x["authors"] != np.nan]
            if isinstance(x["authors"], list)
            else np.nan,
            axis=1,
        ),
    )
    .explode("authors_list")
    .assign(
        authors_list=lambda x: x["authors_list"].str.lower(),
        semantic=lambda x: x.apply(
            lambda x: "".join(
                (
                    c
                    for c in unicodedata.normalize("NFD", x["authors_list"])
                    if unicodedata.category(c) != "Mn"
                )
            ),
            axis=1,
        ),
    )
    .rename(columns={
        "url": "url_paper",
        "externalIds.DBLP": "dblp_paper",
        "externalIds.MAG":"mag_paper",
        "externalIds.DOI":"doi_paper",
    })
    .merge(
        (
            pd.json_normalize(list_paper_semantic, "authors_detail")
            .assign(
                name=lambda x: x["name"].str.lower(),
                semantic=lambda x: x.apply(
                    lambda x: "".join(
                        (
                            c
                            for c in unicodedata.normalize("NFD", x["name"])
                            if unicodedata.category(c) != "Mn"
                        )
                    ),
                    axis=1,
                ),
            )
            .drop_duplicates(subset=["name"])
            .assign(
                total_paper=lambda x: x["papers"].str.len(),
                esg=lambda x: x.apply(lambda x: count_esg(x["papers"]), axis=1),
                pct_esg=lambda x: x["esg"] / x["total_paper"],
            )
            .rename(columns={"url": "url_author",
                             "externalIds.DBLP": "dblp_author"})
        ),
        how="left",
        on=["semantic"],
    )
    .merge(
        df_tsne.drop(columns=["abstract"]).rename(columns={"size": "size_abstract"}),
        how="left",
        on=["paperId"],
    )
    .assign(publication_name=lambda x: x["publication_name"].str.lower())
    .merge(df_publication_rank)
    .reindex(
        columns=[
            "publication_name",
            "rang_digit",
            "paperId",
            "url_paper",
            "title",
            "abstract",
            "venue",
            "year",
            "referenceCount",
            "citationCount",
            "influentialCitationCount",
            "isOpenAccess",
            "fieldsOfStudy",
            "mag_paper",
            "doi_paper",
            "dblp_paper",
            "nb_authors",
             "cluster_w_emb",
            "sentiment",
            "lenght",
            "adj",
            "noun",
            "verb",
            "size_abstract",
            "pct_adj",
            "pct_noun",
            "pct_verb",
            "authors",
            "status",
            "author_details_semantic",
            "name",
            "aliases",
            "authorId",
            "url_author",
            "affiliations",
            "homepage",
            "papers",
            "gender.gender",
            "gender.probability",
            "dblp_author",
            "total_paper",
            "esg",
            "pct_esg"
        ]
    )
    .rename(columns = {
                      "gender.gender":"gender",
                       "gender.probability":"gender_proba"})
)
```

```python
df_authors_journal_full.shape
```

```python
df_authors_journal_full.head(2).reindex(
        columns=[
            "publication_name",
            "rang_digit",
            "title",
             "cluster_w_emb",
            "sentiment",
            "status",
            "name",
            "gender",
            "gender_proba",
            "esg",
            "pct_esg"
        ]
    )
```

```python
FILENAME_SPREADSHEET = "AUTHOR_SEMANTIC_GOOGLE"
df_authors_journal_full.to_csv('AUTHOR_SEMANTIC_GOOGLE.csv', index = False)
drive.upload_file_root(mime_type = 'text/plain',
                 file_name = 'AUTHOR_SEMANTIC_GOOGLE.csv',
                 local_path = "AUTHOR_SEMANTIC_GOOGLE.csv"
                ) 
drive.move_file(file_name = 'AUTHOR_SEMANTIC_GOOGLE.csv', folder_name = "SPREADSHEETS_ESG_METADATA")
```

# Table `meta_analysis_esg_cfp`

Since the table to create has missing value, please use the following at the top of the query

```
CREATE TABLE database.table_name WITH (format = 'PARQUET') AS
```


Choose a location in S3 to save the CSV. It is recommended to save in it the `datalake-datascience` bucket. Locate an appropriate folder in the bucket, and make sure all output have the same format

```python
s3_output = 'DATA/FINANCE/ESG/ESG_CFP'
table_name = 'meta_analysis_esg_cfp'
DatabaseName = 'esg'
s3_output_example = 'SQL_OUTPUT_ATHENA'
```

First, we need to delete the table (if exist)

```python
try:
    response = glue.delete_table(
        database=DatabaseName,
        table=table_name
    )
    print(response)
except Exception as e:
    print(e)
```

Clean up the folder with the previous csv file. Be careful, it will erase all files inside the folder

```python
s3.remove_all_bucket(path_remove = s3_output)
```

```python
%%time
query = """
-- CREATE TABLE {0}.{1} WITH (format = 'PARQUET') AS
WITH merge AS (
  SELECT 
    id, 
    id_old,
    image,
    row_id_excel,
    table_refer,
    row_id_google_spreadsheet,
    incremental_id,
    paper_name, 
    --publication_year, 
    publication_type, 
    regexp_replace(
      regexp_replace(
        lower(publication_name), 
        '\&', 
        'and'
      ), 
      '\-', 
      ' '
    ) as publication_name, 
    cnrs_ranking, 
    UPPER(peer_reviewed) as peer_reviewed, 
    UPPER(study_focused_on_social_environmental_behaviour) as study_focused_on_social_environmental_behaviour, 
    type_of_data, 
    CASE WHEN regions = 'ARAB WORLD' THEN 'WORLDWIDE' ELSE regions END AS regions,
    CASE WHEN study_focusing_on_developing_or_developed_countries = 'Europe' THEN 'WORLDWIDE' ELSE UPPER(study_focusing_on_developing_or_developed_countries) END AS study_focusing_on_developing_or_developed_countries,
    first_date_of_observations,
    last_date_of_observations,
    CASE WHEN first_date_of_observations >= 1997 THEN 'YES' ELSE 'NO' END AS kyoto,
    CASE WHEN first_date_of_observations >= 2009 THEN 'YES' ELSE 'NO' END AS financial_crisis,
    last_date_of_observations - first_date_of_observations as windows,
    adjusted_model_name,
    adjusted_model,
    dependent, 
    adjusted_dependent,
    independent,
    adjusted_independent, 
    social,
    environmental,
    governance,
    sign_of_effect,
    target,
    p_value_significant,
    sign_positive,
    sign_negative,
    lag, 
    interaction_term, 
    quadratic_term, 
    n, 
    r2, 
    beta, 
    to_remove,
    test_standard_error,
    test_p_value,
    test_t_value,
    adjusted_standard_error,
    adjusted_t_value
  FROM 
    esg.papers_meta_analysis_new 
    LEFT JOIN (
      SELECT 
        DISTINCT(title),
        nr as id_old, 
        publication_year, 
        publication_type, 
        publication_name, 
        cnrs_ranking, 
        peer_reviewed, 
        study_focused_on_social_environmental_behaviour, 
        type_of_data, 
        study_focusing_on_developing_or_developed_countries
      FROM 
        esg.papers_meta_analysis
    ) as old on papers_meta_analysis_new.id = old.id_old
    -- WHERE to_remove = 'TO_KEEP'
LEFT JOIN (
SELECT 
        nr,
        CAST(MIN(first_date_of_observations) as int) as first_date_of_observations,
        CAST(MAX(last_date_of_observations)as int) as last_date_of_observations,
        min(row_id_excel) as row_id_excel
      FROM 
        esg.papers_meta_analysis
        GROUP BY nr
) as date_pub on papers_meta_analysis_new.id = date_pub.nr
LEFT JOIN (
SELECT 
  nr, 
  MIN(regions) as regions 
FROM 
  (
    SELECT 
      nr, 
      CASE WHEN regions_of_selected_firms in (
        'Cameroon', 'Egypt', 'Libya', 'Morocco', 
        'Nigeria'
      ) THEN 'AFRICA' WHEN regions_of_selected_firms in ('GCC countries') THEN 'ARAB WORLD' WHEN regions_of_selected_firms in (
        'India', 'Indonesia', 'Taiwan', 'Vietnam', 
        'Australia', 'China', 'Iran', 'Malaysia', 
        'Pakistan', 'South Korea', 'Bangladesh'
      ) THEN 'ASIA AND PACIFIC' WHEN regions_of_selected_firms in (
        'Spain', '20 European countries', 
        'United Kingdom', 'France', 'Germany, Italy, the Netherlands and United Kingdom', 
        'Turkey', 'UK'
      ) THEN 'EUROPE' WHEN regions_of_selected_firms in ('Latin America', 'Brazil') THEN 'LATIN AMERICA' WHEN regions_of_selected_firms in ('USA', 'US', 'U.S.', 'Canada') THEN 'NORTH AMERICA' ELSE 'WORLDWIDE' END AS regions 
    FROM 
      papers_meta_analysis
  ) 
GROUP BY 
  nr
) as reg on papers_meta_analysis_new.id = reg.nr
) 
SELECT 
    to_remove, 
    id, 
    id_old,
    image,
    row_id_excel,
    row_id_google_spreadsheet,
    table_refer,
    incremental_id,
    paper_name,
    --publication_name,
    rank,
    sjr, 
    sjr_best_quartile, 
    h_index, 
    total_docs_2020, 
    total_docs_3years, 
    total_refs, 
    total_cites_3years, 
    citable_docs_3years, 
    cites_doc_2years, 
    country ,
    --publication_year, 
    publication_type, 
    --cnrs_ranking, 
    peer_reviewed, 
    study_focused_on_social_environmental_behaviour, 
    type_of_data, 
    regions,
    region_journal,
    study_focusing_on_developing_or_developed_countries,
    first_date_of_observations,
    last_date_of_observations - (windows/2) as mid_year,
    last_date_of_observations,
    kyoto,
    financial_crisis,
    windows,
    adjusted_model_name,
    adjusted_model,
    dependent, 
    adjusted_dependent,
    independent,
    adjusted_independent, 
    social,
    environmental,
    governance,
    sign_of_effect,
    target,
    p_value_significant,
    sign_positive,
    sign_negative,
    lag, 
    interaction_term, 
    quadratic_term, 
    n, 
    r2, 
    beta, 
    test_standard_error,
    test_p_value,
    test_t_value,
    adjusted_standard_error,
    adjusted_t_value 
FROM 
  merge 
  LEFT JOIN (
    SELECT 
      rank, 
      regexp_replace(
        regexp_replace(
          lower(title), 
          '\&', 
          'and'
        ), 
        '\-', 
        ' '
      ) as title, 
      sjr, 
      sjr_best_quartile, 
      h_index, 
      total_docs_2020, 
      total_docs_3years, 
      total_refs, 
      total_cites_3years, 
      citable_docs_3years, 
      cites_doc_2years, 
      country,
      region as region_journal
    FROM 
      "scimago"."journals_scimago"
    WHERE sourceid not in (16400154787)
  ) as journal on merge.publication_name = journal.title
""".format(DatabaseName, table_name)
output = s3.run_query(
                    query=query,
                    database=DatabaseName,
                    s3_output=s3_output_example,
    filename = "temp"
                )
output.head()

```

```python
output.shape
```

We also want to add csr_20_categories and cfp_4_categories, but the original data are not unique so we will keep the unique values using brute force method. 

```python
query_csr = """
SELECT DISTINCT(csr_20_categories), nr as id_old
FROM "esg"."papers_meta_analysis" 
ORDER BY nr
"""
output_csr = (
    s3.run_query(
                    query=query_csr,
                    database=DatabaseName,
                    s3_output=s3_output_example,
    filename = "temp"
                )
    .drop_duplicates(subset = ['id_old'])
)
output_csr.head()
```

```python
query_csr = """
SELECT DISTINCT(cfp_4_categories
), nr as id_old
FROM "esg"."papers_meta_analysis" 
ORDER BY nr
"""
output_cfp = (
    s3.run_query(
                    query=query_csr,
                    database=DatabaseName,
                    s3_output=s3_output_example,
    filename = "temp"
                )
    .drop_duplicates(subset = ['id_old'])
)
output_cfp.head()
```

```python
output = (
    output
    .merge(output_csr)
    .merge(output_cfp)
)
```

```python
output.shape
```

Use Semantic scholar to find ID

```python
def find_id(paper_name):
    """
    to keep thing simple, assume first results in the best option
    """
    paper_name_clean = (
         paper_name
        .lower()
        .replace("  ", "+")
        .replace(" ", "+")
        .replace("\n", "+")
        .replace(",", "+")
        .replace("–", "")
        .replace("++", "+")
        .replace(":", "")
    )
    url_paper = 'https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields={}'.format(
        paper_name_clean, ",".join(field))
    response_1 = requests.get(url_paper, headers=headers)
    if response_1.status_code == 200:
        response_1 = response_1.json()
        if len(response_1['data']) > 0:
            url_paper = "https://api.semanticscholar.org/graph/v1/paper/{}?fields={}".format(
                response_1['data'][0]['paperId'], ",".join(field_paper))
            response_2 = requests.get(url_paper, headers=headers)
            return {'paper_name': paper_name, 'paperId':response_2.json()['paperId']}
```

```python
find_list = False
if find_list:
    list_ids = []
    failure = []
    for p in tqdm(list(output['paper_name'].unique())):
        time.sleep(8)
        try:
            list_ids.append(find_id(p))
        except:
            failure.append(p)
    with open('MODELS_AND_DATA/list_papers_id.pickle', 'wb') as handle:
        pickle.dump(list_ids, handle)
```

```python
list_ids = pickle.load( open( "MODELS_AND_DATA/list_papers_id.pickle", "rb" ))
```

### Add authors information:

- Add the following information from [AUTHOR_SEMANTIC_GOOGLE](https://docs.google.com/spreadsheets/d/1GrrQBip4qNcDuT_MEG9KhNhfTC3yFsVZUtP8-SvXBL4/edit?usp=sharing)

**Regroup data provider**

- MSCI
- KLD rating
- Thomson
- Bloomberg ESG score
- Other
    - 'BIST ESG score':'OTHER',
    - 'Vigeo score':'OTHER',
    - 'Charity':'OTHER',
    - 'EIRIS':'OTHER',
    - 'Fortune':'OTHER',
    - 'Ibase':'OTHER',
    - 'ISO norms':'OTHER',
    - 'RiskMetrics':'OTHER',
    - 'Surveys':'OTHER',
    - 'Environmental disclosure':'OTHER',
    - 'Disclosure of CSR and GRI':'OTHER'

**Regroup journal region**

- Europe
    - Eastern Europe
    - Western Europe
- Northern America
- Southern America
    - Latin America
- Asia-Pacific
    - Asiatic Region
    - Pacific Region
- Africa-middle East
    - Middle East
    - Africa/Middle East
    - Africa


```python
"referenceCount",
            "citationCount",
            "influentialCitationCount",
```

```python
df_final = (
    df_authors_journal_full.groupby(["paperId"])
    .agg(
        {
            "nb_authors": "max",
            "referenceCount": "sum",
            "citationCount": "sum",
            "influentialCitationCount": "sum",
            "isOpenAccess": "min",
            "total_paper": "sum",
            "esg": "sum",
        }
    )
    .assign(pct_esg=lambda x: x["esg"] / x["total_paper"])
    .reset_index()
    .loc[lambda x: x['esg'] != 0]
    .merge(pd.DataFrame([i for i in list_ids if i]), on=["paperId"])
    .merge(
        (
            df_authors_journal_full.groupby(
                ["paperId"])['gender'].value_counts()
            .unstack(-1)
            .fillna(0)
            .assign(
                total=lambda x: x.sum(axis=1),
                pct_female=lambda x: x['FEMALE']/x['total']
            )
            .reset_index()
            .drop(columns=['total'])
        ), on=['paperId']
    )
    .merge(output, on=["paper_name"])
    .rename(columns={
        #'cited_by.total': 'cited_by_total',
        #'referenceCount': 'reference_count',
        #'citationCount': 'citation_count',
        'isOpenAccess': 'is_open_access',
        #'cited_by.total': 'cited_by_total',
        'FEMALE': 'female',
        'MALE': 'male',
        'UNKNOWN': 'unknown'
    })
    .replace(
        {
            'csr_20_categories': {
                'BIST ESG score': 'OTHER',
                'Vigeo score': 'OTHER',
                'Charity': 'OTHER',
                'EIRIS': 'OTHER',
                'Fortune': 'OTHER',
                'Ibase': 'OTHER',
                'ISO norms': 'OTHER',
                'RiskMetrics': 'OTHER',
                'Surveys': 'OTHER',
                'Environmental disclosure': 'OTHER',
                'Disclosure of CSR and GRI': 'OTHER',
                'Other': 'OTHER',
                'Bloomberg ESG score': 'BLOOMBERG',
                'Thomson': 'THOMSON',
                'KLD rating': 'MSCI',
            },
            'region_journal': {
                'Eastern Europe': 'EUROPE',
                'Western Europe': 'EUROPE',
                'Northern America': 'NORTHERN AMERICA',
            }
        }
    )
    .merge(
        (
            df_authors_journal_full
            .reindex(columns=[
                'paperId',
                "year",
                "publication_name",
                "rang_digit",
                'cluster_w_emb',
                'sentiment',
                'lenght',
                'adj',
                'noun',
                'verb',
                'size_abstract',
                'pct_adj',
                'pct_noun',
                'pct_verb'
            ])
            .drop_duplicates(subset=['paperId'])
            
        )
    )
    .rename(columns = 
            {
                'year':"publication_year",
                "referenceCount":'reference_count',
                "citationCount":'citation_count',
                "influentialCitationCount":'influential_citation_count'
                
            })
)

df_final.shape
```

```python
df_final.head(1)
```

```python
(
    df_final
    .groupby('region_journal')
    .agg(
        {'region_journal':'count'}
    )
    .rename(columns = {'region_journal':'count'})
    .sort_values(by = ["count"])
)
```

```python
(
    df_final
    .groupby('csr_20_categories')
    .agg(
        {'csr_20_categories':'count'}
    )
    .rename(columns = {'csr_20_categories':'count'})
    .sort_values(by = ["count"])
)
```

```python
input_path = 'df_esg_metaanalysis.csv'
df_final.to_csv(input_path, index=False)
# SAVE S3
s3.upload_file(input_path, s3_output)
```

```python
schema = [
    {'Name': 'index', 'Type': 'int', 'Comments': ''},
{'Name': 'paperId', 'Type': 'string', 'Comments': ''},
{'Name': 'nb_authors', 'Type': 'int', 'Comments': ''},
{'Name': 'reference_count', 'Type': 'int', 'Comments': ''},
{'Name': 'citation_count', 'Type': 'int', 'Comments': ''},
{'Name': 'influential_citation_count', 'Type': 'int', 'Comments': ''},
{'Name': 'is_open_access', 'Type': 'boolean', 'Comments': ''},
{'Name': 'total_paper', 'Type': 'int', 'Comments': ''},
{'Name': 'esg', 'Type': 'int', 'Comments': ''},
{'Name': 'pct_esg', 'Type': 'float', 'Comments': ''},
{'Name': 'paper_name', 'Type': 'string', 'Comments': ''},
{'Name': 'female', 'Type': 'float', 'Comments': ''},
{'Name': 'male', 'Type': 'float', 'Comments': ''},
{'Name': 'unknown', 'Type': 'float', 'Comments': ''},
{'Name': 'pct_female', 'Type': 'float', 'Comments': ''},
{'Name': 'to_remove', 'Type': 'string', 'Comments': ''},
{'Name': 'id', 'Type': 'int', 'Comments': ''},
{'Name': 'image', 'Type': 'string', 'Comments': ''},
{'Name': 'row_id_excel', 'Type': 'string', 'Comments': ''},
{'Name': 'row_id_google_spreadsheet', 'Type': 'string', 'Comments': ''},
{'Name': 'table_refer', 'Type': 'string', 'Comments': ''},
{'Name': 'incremental_id', 'Type': 'int', 'Comments': ''},
{'Name': 'publication_name', 'Type': 'string', 'Comments': ''},
{'Name': 'rank', 'Type': 'int', 'Comments': ''},
{'Name': 'sjr', 'Type': 'int', 'Comments': ''},
{'Name': 'sjr_best_quartile', 'Type': 'string', 'Comments': ''},
{'Name': 'h_index', 'Type': 'int', 'Comments': ''},
{'Name': 'total_docs_2020', 'Type': 'int', 'Comments': ''},
{'Name': 'total_docs_3years', 'Type': 'int', 'Comments': ''},
{'Name': 'total_refs', 'Type': 'int', 'Comments': ''},
{'Name': 'total_cites_3years', 'Type': 'int', 'Comments': ''},
{'Name': 'citable_docs_3years', 'Type': 'int', 'Comments': ''},
{'Name': 'cites_doc_2years', 'Type': 'int', 'Comments': ''},
{'Name': 'country', 'Type': 'string', 'Comments': ''},
{'Name': 'publication_year', 'Type': 'int', 'Comments': ''},
{'Name': 'publication_type', 'Type': 'string', 'Comments': ''},
{'Name': 'cnrs_ranking', 'Type': 'int', 'Comments': ''},
{'Name': 'peer_reviewed', 'Type': 'string', 'Comments': ''},
{'Name': 'study_focused_on_social_environmental_behaviour', 'Type': 'string', 'Comments': ''},
{'Name': 'type_of_data', 'Type': 'string', 'Comments': ''},
{'Name': 'regions', 'Type': 'string', 'Comments': ''},
{'Name': 'study_focusing_on_developing_or_developed_countries', 'Type': 'string', 'Comments': ''},
{'Name': 'first_date_of_observations', 'Type': 'int', 'Comments': ''},
{'Name': 'mid_year', 'Type': 'int', 'Comments': ''},
{'Name': 'last_date_of_observations', 'Type': 'int', 'Comments': ''},
{'Name': 'kyoto', 'Type': 'string', 'Comments': ''},
{'Name': 'financial_crisis', 'Type': 'string', 'Comments': ''},
{'Name': 'windows', 'Type': 'int', 'Comments': ''},
{'Name': 'adjusted_model_name', 'Type': 'string', 'Comments': ''},
{'Name': 'adjusted_model', 'Type': 'string', 'Comments': ''},
{'Name': 'dependent', 'Type': 'string', 'Comments': ''},
{'Name': 'adjusted_dependent', 'Type': 'string', 'Comments': ''},
{'Name': 'independent', 'Type': 'string', 'Comments': ''},
{'Name': 'adjusted_independent', 'Type': 'string', 'Comments': ''},
{'Name': 'social', 'Type': 'string', 'Comments': ''},
{'Name': 'environmental', 'Type': 'string', 'Comments': ''},
{'Name': 'governance', 'Type': 'string', 'Comments': ''},
{'Name': 'sign_of_effect', 'Type': 'string', 'Comments': ''},
{'Name': 'target', 'Type': 'string', 'Comments': ''},
{'Name': 'p_value_significant', 'Type': 'string', 'Comments': ''},
{'Name': 'sign_positive', 'Type': 'string', 'Comments': ''},
{'Name': 'sign_negative', 'Type': 'string', 'Comments': ''},
{'Name': 'lag', 'Type': 'string', 'Comments': ''},
{'Name': 'interaction_term', 'Type': 'string', 'Comments': ''},
{'Name': 'quadratic_term', 'Type': 'string', 'Comments': ''},
{'Name': 'n', 'Type': 'int', 'Comments': ''},
{'Name': 'r2', 'Type': 'int', 'Comments': ''},
{'Name': 'beta', 'Type': 'int', 'Comments': ''},
{'Name': 'test_standard_error', 'Type': 'string', 'Comments': ''},
{'Name': 'test_p_value', 'Type': 'string', 'Comments': ''},
{'Name': 'test_t_value', 'Type': 'string', 'Comments': ''},
{'Name': 'adjusted_standard_error', 'Type': 'int', 'Comments': ''},
{'Name': 'adjusted_t_value', 'Type': 'int', 'Comments': ''},
{'Name': 'csr_20_categories', 'Type': 'string', 'Comments': ''},
{'Name': 'cfp_4_categories', 'Type': 'string', 'Comments': ''},    
{'Name': 'cluster_w_emb', 'Type': 'string', 'Comments': ''},
{'Name': 'sentiment', 'Type': 'string', 'Comments': ''},
    {'Name': 'region_journal', 'Type': 'string', 'Comments': ''},
{'Name': 'lenght', 'Type': 'float', 'Comments': ''},
{'Name': 'adj', 'Type': 'float', 'Comments': ''},
{'Name': 'pct_adj', 'Type': 'float', 'Comments': ''},
{'Name': 'noun', 'Type': 'float', 'Comments': ''},
{'Name': 'pct_noun', 'Type': 'float', 'Comments': ''},
{'Name': 'verb', 'Type': 'float', 'Comments': ''},
{'Name': 'pct_verb', 'Type': 'float', 'Comments': ''},
{'Name': 'size_abstract', 'Type': 'float', 'Comments': ''}
]

```

```python
glue = service_glue.connect_glue(client=client)

target_S3URI = os.path.join("s3://",bucket, s3_output)
name_crawler = "crawl-industry-name"
Role = 'arn:aws:iam::468786073381:role/AWSGlueServiceRole-crawler-datalake'
DatabaseName = "esg"
TablePrefix = 'meta_analysis_'  # add "_" after prefix, ex: hello_


glue.create_table_glue(
    target_S3URI,
    name_crawler,
    Role,
    DatabaseName,
    TablePrefix,
    from_athena=False,
    update_schema=schema,
)
```

```python
query_count = """
SELECT COUNT(*) AS CNT
FROM {}.{} 
""".format(DatabaseName, table_name)
output = s3.run_query(
                    query=query_count,
                    database=DatabaseName,
                    s3_output=s3_output_example,
    filename = 'count_{}'.format(table_name)
                )
output
```

# Update Glue catalogue and Github

This step is mandatory to validate the query in the ETL.


## Create or update the data catalog

The query is saved in the S3 (bucket `datalake-london`), but the comments are not available. Use the functions below to update the catalogue and Github



Update the dictionary

- DatabaseName:
- TableName:
- ~TablePrefix:~
- input: 
- filename: Name of the notebook or Python script: to indicate
- Task ID: from Coda
- index_final_table: a list to indicate if the current table is used to prepare the final table(s). If more than one, pass the index. Start at 0
- if_final: A boolean. Indicates if the current table is the final table -> the one the model will be used to be trained
- schema: glue schema with comment
- description: details query objective

**Update schema**

If `automatic = False` in `automatic_update`, then the function returns only the variables to update the comments. Manually add the comment, **then**, pass the new schema (only the missing comment) to the argument `new_schema`. 

To update the schema, please use the following structure

```
schema = [
    {
        "Name": "VAR1",
        "Type": "",
        "Comment": ""
    },
    {
        "Name": "VAR2",
        "Type": "",
        "Comment": ""
    }
]
```

```python
%load_ext autoreload
%autoreload 2
import sys
sys.path.append(os.path.join(parent_path, 'utils'))
import make_toc
import create_schema
import create_report
import update_glue_github
```

The function below manages everything automatically. If the final table comes from more than one query, then pass a list of table in `list_tables` instead of `automatic`

```python
list_input,  schema = update_glue_github.automatic_update(
    list_tables = 'automatic',
    automatic= True,
    new_schema = None, ### override schema
    client = client,
    TableName = table_name,
    query = query)
```

```python
description = """
Create table with journal information, papers and coefficients for the meta analysis
"""
name_json = 'parameters_ETL_esg_metadata.json'
partition_keys = ["id", 'incremental_id']
notebookname = "00_meta_analysis.ipynb"
dic_information = {
    "client":client,
    'bucket':bucket,
    's3_output':s3_output,
    'DatabaseName':DatabaseName,
    'TableName':table_name,
    'name_json':name_json,
    'partition_keys':partition_keys,
    'notebookname':notebookname,
    'index_final_table':[0],
    'if_final': 'True',
    'schema':schema,
    'description':description,
    'query':query,
    "list_input":list_input,
    'list_input_automatic':True
}
```

```python
update_glue_github.update_glue_github(client = client,dic_information = dic_information)
```

## Check Duplicates

One of the most important step when creating a table is to check if the table contains duplicates. The cell below checks if the table generated before is empty of duplicates. The code uses the JSON file to create the query parsed in Athena. 

You are required to define the group(s) that Athena will use to compute the duplicate. For instance, your table can be grouped by COL1 and COL2 (need to be string or varchar), then pass the list ['COL1', 'COL2'] 

```python
update_glue_github.find_duplicates(
    client = client,
    bucket = bucket,
    name_json = name_json,
    partition_keys = partition_keys,
    TableName= table_name
)
```

## Count missing values

```python
update_glue_github.count_missing(client = client, name_json = name_json, bucket = bucket,TableName = table_name)
```

# Update Github Data catalog

The data catalog is available in Glue. Although, we might want to get a quick access to the tables in Github. In this part, we are generating a `README.md` in the folder `00_data_catalogue`. All tables used in the project will be added to the catalog. We use the ETL parameter file and the schema in Glue to create the README. 

Bear in mind the code will erase the previous README. 

```python
create_schema.make_data_schema_github(name_json = name_json)
```

<!-- #region heading_collapsed="true" -->
# Analytics

In this part, we are providing basic summary statistic. Since we have created the tables, we can parse the schema in Glue and use our json file to automatically generates the analysis.

The cells below execute the job in the key `ANALYSIS`. You need to change the `primary_key` and `secondary_key` 
<!-- #endregion -->

For a full analysis of the table, please use the following Lambda function. Be patient, it can takes between 5 to 30 minutes. Times varies according to the number of columns in your dataset.

Use the function as follow:

- `output_prefix`:  s3://datalake-datascience/ANALYTICS/OUTPUT/TABLE_NAME/
- `region`: region where the table is stored
- `bucket`: Name of the bucket
- `DatabaseName`: Name of the database
- `table_name`: Name of the table
- `group`: variables name to group to count the duplicates
- `primary_key`: Variable name to perform the grouping -> Only one variable for now
- `secondary_key`: Variable name to perform the secondary grouping -> Only one variable for now
- `proba`: Chi-square analysis probabilitity
- `y_var`: Continuous target variables

Check the job processing in Sagemaker: https://eu-west-3.console.aws.amazon.com/sagemaker/home?region=eu-west-3#/processing-jobs

The notebook is available: https://s3.console.aws.amazon.com/s3/buckets/datalake-datascience?region=eu-west-3&prefix=ANALYTICS/OUTPUT/&showversions=false

Please, download the notebook on your local machine, and convert it to HTML:

```
cd "/Users/thomas/Downloads/Notebook"
aws s3 cp s3://datalake-datascience/ANALYTICS/OUTPUT/asif_unzip_data_csv/Template_analysis_from_lambda-2020-11-22-08-12-20.ipynb .

## convert HTML no code
jupyter nbconvert --no-input --to html Template_analysis_from_lambda-2020-11-21-14-30-45.ipynb
jupyter nbconvert --to html Template_analysis_from_lambda-2020-11-22-08-12-20.ipynb
```

Then upload the HTML to: https://s3.console.aws.amazon.com/s3/buckets/datalake-datascience?region=eu-west-3&prefix=ANALYTICS/HTML_OUTPUT/

Add a new folder with the table name in upper case

```python
import boto3

key, secret_ = con.load_credential()
client_lambda = boto3.client(
    'lambda',
    aws_access_key_id=key,
    aws_secret_access_key=secret_,
    region_name = region)
```

```python
primary_key = ''
secondary_key = ''
y_var = ''
```

```python
payload = {
    "input_path": "s3://datalake-datascience/ANALYTICS/TEMPLATE_NOTEBOOKS/template_analysis_from_lambda.ipynb",
    "output_prefix": "s3://datalake-datascience/ANALYTICS/OUTPUT/{}/".format(table_name.upper()),
    "parameters": {
        "region": "{}".format(region),
        "bucket": "{}".format(bucket),
        "DatabaseName": "{}".format(DatabaseName),
        "table_name": "{}".format(table_name),
        "group": "{}".format(','.join(partition_keys)),
        "keys": "{},{}".format(primary_key,secondary_key),
        "y_var": "{}".format(y_var),
        "threshold":0
    },
}
payload
```

```python
#response = client_lambda.invoke(
#    FunctionName='RunNotebook',
#    InvocationType='RequestResponse',
#    LogType='Tail',
#    Payload=json.dumps(payload),
#)
#response
```

# Generation report

```python
import os, time, shutil, urllib, ipykernel, json
from pathlib import Path
from notebook import notebookapp
```

```python
create_report.create_report(extension = "html", keep_code = True, notebookname =  notebookname)
```

```python
create_schema.create_schema(name_json, path_save_image = os.path.join(parent_path, 'utils'))
```

```python
### Update TOC in Github
for p in [parent_path,
          str(Path(path).parent),
          os.path.join(str(Path(path).parent), "00_download_data"),
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
