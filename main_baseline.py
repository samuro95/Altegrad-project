import sys
import json
import numpy as np
from model_helper import HAN, HAN_learning
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt

is_GPU = True
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = hyper-parameters = = = = =


drop_rate = 0.5
batch_size = 100
nb_epochs = 40
my_optimizer = 'adam'
my_patience = 8
method = 'linear'
n_units= 32
rnn = 'lstm'
n_dense = 0
n_rnn = 2
n_context_vect = 1

# = = = = = data loading = = = = =

#docs = np.load(path_to_data + 'documents.npy')
docs = np.load(path_to_data + 'documents_025_4.npy')

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

MSE_tab = []

for tgt in range(4):

    if tgt == 3 :
        embeddings = np.load(path_to_data + 'new_embeddings.npy')
    else :
        embeddings = np.load(path_to_data + 'embeddings.npy')
        embeddings[:,5] = embeddings[:,5]/np.linalg.norm(embeddings[:,5])
        embeddings[:,6] = embeddings[:,6]/np.linalg.norm(embeddings[:,6])

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()


    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    tab = []
    n_units_tab = [32]
    for n_units in n_units_tab :
        docs_train = docs[train_idxs_new,:,:]
        docs_val = docs[val_idxs,:,:]
        model = HAN(docs_train, target_train, embeddings, n_units, n_dense, n_rnn, drop_rate, n_context_vect, rnn = rnn , method = method, is_GPU = is_GPU)
        model.summary()
        score, history = HAN_learning(tgt, model, docs_train, target_train, docs_val, target_val, my_optimizer, my_patience, nb_epochs, batch_size, path_to_data)
        tab.append(score)
        plt.figure(tgt)
        plt.plot(history.history['loss'],)
        plt.plot(history.history['val_loss'])
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('normal'+str(tgt)+'.png')

    MSE_tab.append(tab)

print(MSE_tab)
