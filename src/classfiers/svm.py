import time
import asyncio
import logging
import nltk

if __name__ == '__main__':
    import os
    import sys
    logging.getLogger().setLevel(logging.INFO)
    sys.path.append(os.path.dirname(sys.path[0]))


import numpy as np
import pulsar.api as pulsar
from sklearn import svm
from sklearn import metrics

import data_provider

from utils import split


def run_svm_instance(actor, args={}):
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='categorical',
                                  stemmer=nltk.stem.lancaster.LancasterStemmer())
    model = create_model(len(X[0]), len(Y[0]))
    asyncio.ensure_future(svm_instance_process(actor, args))


async def svm_instance_process(actor, args={}):
    for i in range(200001):
        # actor.logger.info(str(i))
        model = i
        for j in range(1000):
            z = 6 * j + j * j
        if i % 10000 == 0:
            await pulsar.send('svm_manager_actor', 'svm_progress_update', {'aid': actor.aid,
                                                                           'timestamp': time.time(),
                                                                           'progress': 100 * i / 200001})
            await pulsar.send('svm_manager_actor', 'svm_model_update', model)
    await pulsar.send('svm_manager_actor', 'svm_progress_update', {'aid': actor.aid,
                                                                   'timestamp': time.time(),
                                                                   'progress': 101})
    return False


def create_model():
    return svm.LinearSVC(C=1e3, max_iter=10000)


def local_train(args={}):
    model = args['model']
    X_train, X_val = split(args['X'], 0.9)
    Y_train, Y_val = split(args['Y'], 0.9)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    model.fit(X_train, Y_train)
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
    return model.predict(X)


if __name__ == '__main__':
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='numerical',
                                  stemmer=nltk.stem.lancaster.LancasterStemmer())
    model = create_model()
    local_train({'X': X, 'Y': Y, 'model': model})
