import sys
import json
import numpy as np
from model_helper import HAN, HAN_learning


is_GPU = False
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = hyper-parameters = = = = =

n_context_vect = 3
n_units = 50
drop_rate = 0.5
batch_size = 96
nb_epochs = 10
my_optimizer = 'adam'
my_patience = 4


# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]

# create validation set

np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new,:,:]
docs_val = docs[val_idxs,:,:]

for tgt in range(4):

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()

    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    print('data loaded')

    model = HAN(docs_train, target_train, embeddings, n_units, drop_rate,n_context_vect, is_GPU = False)

    HAN_learning(tgt, model, docs_train, target_train, docs_val, target_val, my_optimizer, my_patience, nb_epochs, batch_size, path_to_data)
