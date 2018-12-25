import asyncio
import time
import logging

import tflearn
import numpy as np
import pulsar.api as pulsar

from sklearn import metrics

if __name__ == '__main__':
    import sys
    import os
    logging.getLogger().setLevel(logging.INFO)
    sys.path.append(os.path.dirname(sys.path[0]))

import data_provider

from utils import split, pad_sequences


def run_network_instance(actor, args={}):
    actor.logger.info("Hello")
    X, Y = zip(*data_provider.get_data('../scrapper/out/unijokes.json'))
    pad_sequences(X)
    actor.logger.info(args)
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
    return net


def local_train(args={}):
    model = tflearn.DNN(args['model'], tensorboard_verbose=0)
    X_train, X_val = split(args['X'], 0.9)
    Y_train, Y_val = split(args['Y'], 0.9)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    for i in range(20):
        model.fit(X_train, Y_train, n_epoch=1, show_metric=True)
        compute_metrics(model, X_val, Y_val)


def compute_metrics(model, X, Y):
    predictions = model.predict(X)
    Y_pred = np.zeros(shape=(len(Y), len(Y[0])))
    for prediction, y in zip(predictions, Y_pred):
        y[prediction.argmax()] = 1
    assert(all(sum(x) > 0 for x in Y_pred))

    precision = 100*metrics.precision_score(Y, Y_pred, average="weighted") 
    recall = 100*metrics.recall_score(Y, Y_pred, average="weighted") 
    accuracy = 100*metrics.accuracy_score(Y, Y_pred)

    logging.info("Precision: {:.2f}%".format(precision))
    logging.info("Recall: {:.2f}%".format(recall))
    logging.info("Accuracy: {:.2f}%".format(accuracy))

    return precision, recall, accuracy


if __name__ == '__main__':
    X, Y = zip(*data_provider.get_data('../scrapper/out/unijokes.json'))
    X = [data_provider.to_hot_vector(x) for x in X]
    model = create_model(len(X[0]), len(Y[0]))
    local_train({'X': X, 'Y':Y, 'model':model})
