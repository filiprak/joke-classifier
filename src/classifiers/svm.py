import time
import asyncio
import nltk
import gc

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import pulsar.api as pulsar
from sklearn import svm

import data_provider

from utils import split, compute_metrics


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


def create_model(C, max_iter):
    return svm.LinearSVC(C=C, max_iter=max_iter)


def local_train(stemmer=data_provider.NoStemmer(), text_representation='bag-of-words', C=1e3, max_iter=10000):
    data_provider.STATE = data_provider.initial_state()
    data_provider.STATE['stemmer'] = stemmer
    X, Y = data_provider.get_data(input_format='hot_vector',
                                  output_format='numerical',
                                  ngrams=text_representation=='ngrams',
                                  all_data=True)
    model = create_model(C, max_iter)
    X_train, X_val = split(X, 0.9)
    Y_train, Y_val = split(Y, 0.9)

    data_provider.STATE = data_provider.initial_state()
    del X, Y
    gc.collect()
    #X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    data_provider.STATE = data_provider.initial_state()
    print(">>> {} {} {} {} {}".format(type(stemmer).__name__, text_representation, 'svm', C, max_iter))
    start = time.time()
    model.fit(X_train, Y_train)
    print(">>> TRAINING TIME: {}s".format(time.time() - start))
    Y_pred = model.predict(X_val)
    compute_metrics(Y_val, Y_pred)


if __name__ == '__main__':
    local_train(text_representation='ngrams', max_iter=100000)
