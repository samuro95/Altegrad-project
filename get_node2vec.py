from node2vec import Node2Vec
import networkx as nx
import os
import re

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

for idx,edgelist in enumerate(edgelists):
    g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist) # construct graph from edgelist
    node2vec = Node2Vec(g, dimensions=128, walk_length=20, num_walks=100, p=p, q=q, workers=workers)
    model = node2vec.fit(window=10, min_count=1)
    model.wv.save_word2vec_format(path_to_data+'node2vec_emb/node2vec_emb_'+edgelist)


