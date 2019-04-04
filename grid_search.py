import sys
import json
import numpy as np
from model_helper import HAN, HAN_learning, logistic_regression, get_doc_embedding
from preprocessing_baseline import main as preprocessing


is_GPU = False
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = preprocessing_baseline - parameters = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

num_walks = 5
walk_length = 10
max_doc_size = 70 # maximum number of 'sentences' (walks) in each pseudo-document

node2vec = True

embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    train_idxs = [int(elt) for elt in train_idxs]

train_size_ration = 0.5
train_idxs = train_idxs[:int(len(train_idxs)*train_size_ration)]

np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)
train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

# = = = = = HAN parameters = = = = =

n_units = 25
drop_rate = 0.5
batch_size = 96
nb_epochs = 5
my_optimizer = 'adam'
my_patience = 4

# grid-search parameters
p_values = [0.25,0.5,1.,2.,4.]
q_values = [0.25,0.5,1.,2.,4.]
p_grid, q_grid  = np.meshgrid(p_values, q_values)
grid_score = np.zeros(p_grid.shape)
for i in range(len(p_grid)):
    for j in range(len(p_grid[0])):
        p,q = p_grid[i,j], q_grid[i,j]
        preprocessing(p = p, q = q)
        docs = np.load(path_to_data + 'documents.npy')
        docs_train = docs[train_idxs_new,:,:]
        docs_val = docs[val_idxs,:,:]

        scores = []
        for tgt in range(4):
            with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
                target = file.read().splitlines()
            target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
            target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')
            model = HAN(docs_train, target_train, embeddings, n_units, drop_rate, is_GPU = False)
            score = HAN_learning(tgt, model, docs_train, target_train, docs_val, target_val, my_optimizer, my_patience, nb_epochs, batch_size, path_to_data, save_weights = False, save_history = False)
            print(score)
            scores.append(score)

        grid_score[i,j] = np.mean(np.array(scores))
