import asyncio
import time
import logging
import nltk

import tflearn
import numpy as np
import pulsar.api as pulsar

from sklearn import metrics

if __name__ == '__main__':
    import os
    import sys
    logging.getLogger().setLevel(logging.INFO)
    sys.path.append(os.path.dirname(sys.path[0]))

import data_provider

from utils import split, pad_sequences


def run_network_instance(actor, args={}):
    actor.logger.info(args)
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='categorical',
                                  stemmer=nltk.stem.lancaster.LancasterStemmer())
    model = create_model(len(X[0]), len(Y[0]))
    asyncio.ensure_future(network_instance_process(actor, dict(args, **{"model":model, "X":X, "Y":Y})))


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
        precision, recall, accuracy = compute_metrics(model, X_val, Y_val)
        await pulsar.send('network_manager_actor', 
                          'network_progress_update', 
                          {'aid': actor.aid,
                           'timestamp': time.time(),
                           'progress': (i+1) * 10,
                           'precision': precision,
                           'recall': recall,
                           'accuracy': accuracy})
    await pulsar.send('network_manager_actor', 
                      'network_progress_update', 
                      {'aid': actor.aid,
                       'timestamp': time.time(),
                       'progress': 101})


def create_model(input_length, output_length, activation='relu'):
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


def local_train(args={}):
    model = args['model']
    X_train, X_val = split(args['X'], 0.9)
    Y_train, Y_val = split(args['Y'], 0.9)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    for i in range(40):
        model.fit(X_train, Y_train, n_epoch=1, show_metric=True)
        Y_pred = get_predictions(model, X_val)
        compute_metrics(Y_val, Y_pred)


def compute_metrics(Y, Y_pred):
    precision = 100*metrics.precision_score(Y, Y_pred, average="weighted") 
    recall = 100*metrics.recall_score(Y, Y_pred, average="weighted") 
    accuracy = 100*metrics.accuracy_score(Y, Y_pred)

    logging.info("Precision: {:.2f}%".format(precision))
    logging.info("Recall: {:.2f}%".format(recall))
    logging.info("Accuracy: {:.2f}%".format(accuracy))

    return precision, recall, accuracy


def get_predictions(model, X):
    predictions = model.predict(X)
    Y_pred = np.zeros(shape=predicions.shape))
    for prediction, y in zip(predictions, Y_pred):
        y[prediction.argmax()] = 1
    assert(all(sum(x) > 0 for x in Y_pred))
    return Y_pred


if __name__ == '__main__':
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='categorical',
                                  stemmer=nltk.stem.lancaster.LancasterStemmer())
    model = create_model(len(X[0]), len(Y[0]))
    local_train({'X': X, 'Y': Y, 'model': model})
