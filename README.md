# Altegrad-project
This project was realized for the Advanced learning for text and graph data (ALTEGRAD) MVA class.
Lecturer : Michalis Vazirgiannis (Polytechnique)

The goal of the data challenge is to implement a graph
regression  problem  with  the  constraint  of  using  a  Hierar-
chical Attention Network (HAN) architecture. 

The dataset is made up of 93,719 undirected,
unweighted graphs.  Each graph is associated to four target
values.  We build a regression model for each of the target
variable. 

You can find the written report of the experiment in the file report.pdf 

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
