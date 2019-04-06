import os
import re
import random
import numpy as np
import networkx as nx
from time import time
from scipy.io import loadmat
from gensim.models import Word2Vec

path_root = '..'
path_to_data = path_root + '/data/'

# = = = = = = = = = = = = = = =

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


embeddings = np.load(path_to_data + 'embeddings.npy')
embeddings[:,5] = embeddings[:,5]/np.linalg.norm(embeddings[:,5])
embeddings[:,6] = embeddings[:,6]/np.linalg.norm(embeddings[:,6])

start_time = time()
edgelists = os.listdir(path_to_data + 'edge_lists/')
edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!
nmax = 29
embedding_adj = np.ones((embeddings.shape[0],nmax))*2
for idx,edgelist in enumerate(edgelists):
    g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist) # construct graph from edgelist
    nodes = g.nodes()
    adj = nx.to_numpy_matrix(g)
    #adj /= np.linalg.norm(adj)
    for i,node in enumerate(nodes) :
        col = np.ravel(adj[:,i])
        col = np.pad(col,(0,nmax-len(col)),'constant', constant_values=0)
        embedding_adj[int(node),:] = col
    if idx % round(len(edgelists)/10) == 0:
        print(idx)

embeddings = np.hstack((embeddings,embedding_adj))

np.save(path_to_data + 'new_embeddings.npy', embeddings, allow_pickle=False)
