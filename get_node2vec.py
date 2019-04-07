from node2vec import Node2Vec
import networkx as nx
import os
import re
import numpy as np
from tqdm import tqdm
import sklearn

path_to_data = '../data/'
p = 1
q = 1
workers = 24

directory = path_to_data +  'node2vec_emb_1_1'

try:
    os.stat(directory)
except:
    os.mkdir(directory) 

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

edgelists = os.listdir(path_to_data + 'edge_lists/')
edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!

done_edgelists = os.listdir('../data/node2vec_emb_1_1/')
done_edgelists = [d.split('_')[-1] for d in done_edgelists]
edgelists = [edg for edg in edgelists if edg not in done_edgelists]

for idx,edgelist in enumerate(edgelists):
    g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist) # construct graph from edgelist
    node2vec = Node2Vec(g, dimensions=20, walk_length=15, num_walks=10, p=p, q=q, workers=workers)
    model = node2vec.fit(window=10, min_count=1)
    model.wv.save_word2vec_format(path_to_data+'node2vec_emb_1_1/node2vec_emb_'+edgelist)

fls = os.listdir(directory)

embs = []
for f in tqdm(range(len(fls))): 
    embs.append(np.genfromtxt(data_dir+fls[f], skip_header=1))

embs = np.concatenate(embs, axis=0)
embs = embs[embs[:,0].argsort()]

from sklearn.decomposition import PCA

pca = PCA(n_components=20)

principalComponents = pca.fit_transform(embs_short)

add = np.zeros((1,20))
pca = np.vstack((principalComponents, add))

np.save('../data/new_embeddings.npy', pca)
