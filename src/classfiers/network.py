import asyncio
import time
import nltk
import logging

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


def local_train(args={}):
    model = args['model']
    models = []
    for i in range(40):
        X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                      input_format='hot_vector',
                                      output_format='categorical')
        if not X or not Y:
            logging.info("Out of data")
            break
        X_train, X_val = split(X, 0.9)
        Y_train, Y_val = split(Y, 0.9)
        X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
        model.fit(X_train, Y_train, n_epoch=10, show_metric=True)
        Y_pred = get_predictions(model, X_val)
        compute_metrics(Y_val, Y_pred)

        # This is for demonstration on how to serialize/agregate/update the model
        models.append(serialize_model(model))
        if i % 10 == 0 and i != 0:
            averaged = average_models(models)
            update_model(model, averaged)
            # After that, the accuracy of the model should go down (since we've aggregated models from early stages of training)


def get_predictions(model, X):
    predictions = model.predict(X)
    Y_pred = np.zeros(shape=predictions.shape)
    for prediction, y in zip(predictions, Y_pred):
        y[prediction.argmax()] = 1
    assert(all(sum(x) > 0 for x in Y_pred))
    return Y_pred


if __name__ == '__main__':
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='categorical')
    model = create_model(len(X[0]), len(Y[0]))
    local_train({'model': model})
