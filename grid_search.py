import sys
import json
import numpy as np
from model_helper import HAN, HAN_learning
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import ParameterGrid

is_GPU = True
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)


# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')
embeddings[:,5] = embeddings[:,5]/np.linalg.norm(embeddings[:,5])
embeddings[:,6] = embeddings[:,6]/np.linalg.norm(embeddings[:,6])

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

s = int(len(train_idxs)/2)
train_idxs = [int(elt) for elt in train_idxs]

# create validation set

np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new,:,:]
docs_val = docs[val_idxs,:,:]

res = []

for tgt in range(4):

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()

    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    print('data loaded')

    # = = = = = hyper-parameters = = = = =

    n_context_vect = 1
    drop_rate = 0.5
    batch_size = 100
    nb_epochs = 20
    my_optimizer = 'adam'
    my_patience = 4
    n_dense = 0
    method = 'linear'
    n_units_grid = [32,48,64]
    grid = {'n_units': n_units_grid}
    grid = ParameterGrid(grid)

    best_param = grid[0]
    best_score = 10

    for param in grid :
        n_units = param['n_units']
        print(param)
        model = HAN(docs_train, target_train, embeddings, n_units, n_dense, drop_rate, n_context_vect, rnn = 'lstm', method  = method, is_GPU = is_GPU)
        model.summary()
        score = HAN_learning(tgt, model, docs_train, target_train, docs_val, target_val, my_optimizer, my_patience, nb_epochs, batch_size, path_to_data)
        print(tgt,param,score)
        if score < best_score :
            best_param = param
            best_score = score
        res.append((tgt,param,score))
    print(tgt,best_param, best_score)

print(res)
