import asyncio
import time
import nltk
import logging
import gc

import tflearn
import numpy as np
import pulsar.api as pulsar

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))

import data_provider

from utils import split, pad_sequences, compute_metrics, update_model, average_models, serialize_model


def run_network_instance(actor, args={}):
    asyncio.ensure_future(network_instance_process(actor, args))


async def network_instance_process(actor, args={}):
    model = tflearn.DNN(args['model'], tensorboard_verbose=0)
    X_train, X_val = split(args['X'], 0.9)
    Y_train, Y_val = split(args['Y'], 0.9)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    await pulsar.send('network_manager_actor', 
                      'network_progress_update', 
                      {'aid': actor.aid,
                       'timestamp': time.time(),
                       'progress': 1})
    for i in range(10):
        model.fit(X_train, Y_train, n_epoch=1, show_metric=True)
        Y_pred = get_predictions(model, X_val)
        precision, recall, accuracy = compute_metrics(Y_val, Y_pred)
        await pulsar.send('network_manager_actor',
                          'network_progress_update', 
                          {'aid': actor.aid,
                           'timestamp': time.time(),
                           'progress': (i+1) * 10,
                           'precision': precision,
                           'recall': recall,
                           'accuracy': accuracy})

    await pulsar.send('network_manager_actor', 'network_model_update', serialize_model(model))
    await pulsar.send('network_manager_actor', 
                      'network_progress_update', 
                      {'aid': actor.aid,
                       'timestamp': time.time(),
                       'progress': 101})


<<<<<<< HEAD
def dnn_model(input_length, output_length, activation='relu'):
    input_layer = tflearn.input_data(shape=[None, input_length])
    model = tflearn.fully_connected(input_layer, 
                                    64, 
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    model = tflearn.fully_connected(input_layer, 
                                    64, 
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    softmax = tflearn.fully_connected(model, output_length, activation='softmax')
    sgd = tflearn.SGD(learning_rate=0.1, decay_step=1000)
    net = tflearn.regression(softmax, 
                             optimizer=sgd,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


def lstm_model(input_length, output_length, activation='relu'):
    input_layer = tflearn.input_data(shape=[None, input_length])
    model = tflearn.embedding(input_layer, input_dim=data_provider.STATE['tokenizer'].index, output_dim=128)
    model = tflearn.lstm(model, 128, dropout=0.8)
    softmax = tflearn.fully_connected(model, output_length, activation='softmax')
    sgd = tflearn.SGD(learning_rate=0.1, decay_step=1000)
    net = tflearn.regression(softmax, 
                             optimizer=sgd,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model


def local_train(stemmer=data_provider.NoStemmer(), text_representation='bag-of-words', create_model=dnn_model):
    data_provider.STATE = data_provider.initial_state()
    data_provider.STATE['stemmer'] = stemmer
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector' if create_model == dnn_model else 'sequential',
                                  output_format='categorical',
                                  ngrams=text_representation=='ngrams',
                                  all_data=True)
    model = create_model(len(X[0]), len(Y[0]))
    X_train, X_val = split(X, 0.9)
    Y_train, Y_val = split(Y, 0.9)

    data_provider.STATE = data_provider.initial_state()
    del X, Y
    gc.collect()

    run_id = get_run_id(stemmer, text_representation, create_model)
    print(run_id)
    for i in range(40):
        model.fit(X_train, Y_train, n_epoch=5, validation_set=(X_val, Y_val), show_metric=True, run_id=run_id)
        Y_pred = get_predictions(model, X_val)
        compute_metrics(Y_val, Y_pred)


def get_run_id(stemmer, text, model):
    return "{} {} {}".format(
        type(stemmer).__name__,
        text,
        model.__name__
    )


def get_predictions(model, X):
    predictions = model.predict(X)
    Y_pred = np.zeros(shape=predictions.shape)
    for prediction, y in zip(predictions, Y_pred):
        y[prediction.argmax()] = 1
    assert(all(sum(x) > 0 for x in Y_pred))
    return Y_pred


if __name__ == '__main__':
    local_train(create_model=lstm_model)
