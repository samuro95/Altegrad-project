# Altegrad-project

## Pre-processing

In order to reproduce our results, firstly run `preprocessing_baseline.py` using the parameters: 

- `node2vec = False, p = 0.25, q = 4` (save as `documents.npy`)
- `node2vec = True, p = 0.25, q = 4` (save as `documents_025_4.npy`)
- `node2vec = True, p = 4, q = 0.25` (save as `documents_025_4.npy`)

with all other parameters as given. 

Run `get_node2vec.py` to get `new_embeddings.npy`. (node2vec embeddings, takes a number of hours)

## Model

Run `main.py`.

Our final choices for the model are in `model_helper.py`. 

## Predictions

To obtain predictions, run `read_results_predict.py`. 
