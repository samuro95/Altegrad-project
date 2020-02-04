# Altegrad-project
The goal of the data challenge is to implement a graph
regression  problem  with  the  constraint  of  using  a  Hierar-
chical Attention Network (HAN) architecture. The graph is
first represented as a document whose words are the nodes
and whose sentences are random walks sampled from the
network. Then, the documents are vectorized as in standard
NLP algorithms.  The HAN is build on top of this embed-
ding layer.   The dataset is made up of 93,719 undirected,
unweighted graphs.  Each graph is associated to four target
values.  We build a regression model for each of the target
variable. A basic HAN implemented in keras is given. The
objective  is  to  optimize  the  different  levels  of  the  overall
architecture, from the random walks to the final regression
layer.  
The goal of the data challenge is to implement a graph
regression  problem  with  the  constraint  of  using  a  Hierar-
chical Attention Network (HAN) architecture. The graph is
first represented as a document whose words are the nodes
and whose sentences are random walks sampled from the
network. Then, the documents are vectorized as in standard
NLP algorithms.  The HAN is build on top of this embed-
ding layer.   The dataset is made up of 93,719 undirected,
unweighted graphs.  Each graph is associated to four target
values.  We build a regression model for each of the target
variable. A basic HAN implemented in keras is given. The
objective  is  to  optimize  the  different  levels  of  the  overall
architecture, from the random walks to the final regression
layer.  

## Pre-processing

In order to reproduce our results, firstly run `preprocessing_baseline.py` using the parameters: 

- `node2vec = False, p = 0.25, q = 4` (save as `documents.npy`)
- `node2vec = True, p = 0.25, q = 4` (save as `documents_025_4.npy`)
- `node2vec = True, p = 4, q = 0.25` (save as `documents_4_025.npy`)

with all other parameters as given. 

Run `get_node2vec.py` to get `new_embeddings.npy`. (node2vec embeddings, takes a number of hours)

## Model

Run `main.py`.

Our final choices for the model are in `model_helper.py`. 

## Predictions

To obtain predictions, run `read_results_predict.py`. 
