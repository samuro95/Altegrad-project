import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Lambda, BatchNormalization, GaussianNoise, CuDNNLSTM,LSTM, Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense
from AttentionWithContext import AttentionWithContext
from keras import regularizers
from sklearn.metrics import mean_squared_error
from keras import backend as K

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh',
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

def bidir_lstm(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNLSTM(units=n_units,
                                      return_sequences = True),
                             merge_mode='concat', weights = None)(my_seq)
    else:
        return Bidirectional(LSTM(units=n_units,
                                     activation='tanh',
                                     dropout=0.0,
                                     recurrent_dropout=0.0,
                                     implementation=1,
                                     return_sequences=True,
                                     recurrent_activation='sigmoid'),
                                 merge_mode='concat', weights=None)(my_seq)

def embed(doc,embeddings):
    x_emb = []
    for seq in doc :
        x_emb.append([embeddings[node,:] for node in seq])
    return(np.array(x_emb))

def get_doc_embedding(docs_train,docs_val,embeddings):
    x_train = np.array([embed(doc_train,embeddings).flatten() for doc_train in docs_train])
    x_val = np.array([embed(doc_val,embeddings).flatten() for doc_val in docs_val])
    return(x_train,x_val)

def logistic_regression(x_train, target_train, x_val, target_val, embeddings, C = None):
    logreg = LogisticRegression()
    clf = logreg.fit(x_train, target_train)
    pred_val = clf.predict(x_val)
    score = mean_squared_error(pred_val, target_val)
    print('logreg done')
    return(score)

def HAN(docs_train, target_train, embeddings, n_units, n_dense, n_rnn, drop_rate, n_context_vect, rnn = 'gru', method = 'sigmoid', is_GPU = True):

    if method == 'sigmoid' :
        min = np.min(target_train)
        max = np.max(target_train)
        d = np.abs(max - min)
        target_train = (target_train + abs(min))/d

    # = = = = = defining architecture = = = = =

    sent_ints = Input(shape=(docs_train.shape[2],))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable= False,
                        )(sent_ints)

    #sent_wv = GaussianNoise(stddev=0.3)(sent_wv)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = sent_wv_dr
    if rnn == 'gru' :
        for k in range(n_rnn) :
            sent_wa = bidir_gru(sent_wa,n_units,is_GPU)
    if rnn == 'lstm' :
        for k in range(n_rnn) :
            sent_wa = bidir_lstm(sent_wa,n_units,is_GPU)
    sent_att_vec,word_att_coeffs = AttentionWithContext(n_context_vect = n_context_vect, return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)

    sent_encoder = Model(sent_ints,sent_att_vec_dr)

    sent_encoder.summary()

    doc_ints = Input(shape=(docs_train.shape[1],docs_train.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = sent_att_vecs_dr
    if rnn == 'gru' :
        for k in range(n_rnn) :
            doc_sa = bidir_gru(doc_sa,n_units,is_GPU)
    if rnn == 'lstm' :
        for k in range(n_rnn) :
            doc_sa = bidir_lstm(doc_sa,n_units,is_GPU)
    doc_att_vec,sent_att_coeffs = AttentionWithContext(n_context_vect = n_context_vect, return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    #doc_att_vec_dr = BatchNormalization()(doc_att_vec_dr)

    for k in range(n_dense) :
         doc_att_vec_dr = Dense(units = int(n_units/2.), activation='relu')(doc_att_vec_dr)
         doc_att_vec_dr = Dropout(0.2)(doc_att_vec_dr)

    #doc_att_vec_dr = BatchNormalization()(doc_att_vec_dr)

    if method == 'sigmoid' :
        preds = Dense(units=1, activation='sigmoid')(doc_att_vec_dr)
        preds = Lambda(lambda x: x*d - abs(min))(preds)
    else :
        doc_att_vec_dr = Dense(units = int(n_units/2.), activation='relu')(doc_att_vec_dr)
        preds = Dense(units=1, activation='linear')(doc_att_vec_dr)

    model = Model(doc_ints,preds)

    print(sent_encoder.summary)

    return(model)


def HAN_learning(tgt, model, docs_train, target_train, docs_val, target_val, my_optimizer, my_patience, nb_epochs, batch_size, path_to_data, save_weights= True, save_history = True):

    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer,
                  metrics=['mae'])

    print('model compiled')

    # = = = = = training = = = = =

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=my_patience,
                                   mode='min')

    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_' + str(tgt),
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

    if save_weights:
        my_callbacks = [early_stopping,checkpointer]
    else:
        my_callbacks = [early_stopping]

    history = model.fit(docs_train,
              target_train,
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (docs_val,target_val),
              callbacks = my_callbacks)

    hist = model.history.history

    if save_history:
        with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)

    score = model.evaluate(docs_val, target_val)

    val_mse = hist['val_loss']
    min_val_mse = min(val_mse)
    best_epoch = val_mse.index(min_val_mse) + 1

    print('* * * * * * * target',tgt,'done * * * * * * *')

    return(min_val_mse,history)
