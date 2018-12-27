import time
import asyncio
import nltk

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import pulsar.api as pulsar
from sklearn import naive_bayes

import data_provider

from utils import split, compute_metrics


def run_bayes_instance(actor, args={}):
    asyncio.ensure_future(bayes_instance_process(actor, args))


async def bayes_instance_process(actor, args={}):
    for i in range(200001):
        # actor.logger.info(str(i))
        model = i
        for j in range(1000):
            z = 6 * j + j * j
        if i % 10000 == 0:
            await pulsar.send('bayes_manager_actor', 'bayes_progress_update', {'aid': actor.aid,
                                                                               'timestamp': time.time(),
                                                                                'progress': 100 * i / 200001})
            await pulsar.send('bayes_manager_actor', 'bayes_model_update', model)
    await pulsar.send('bayes_manager_actor', 'bayes_progress_update', {'aid': actor.aid,
                                                                       'timestamp': time.time(),
                                                                        'progress': 101})
    return False


def create_model():
    return naive_bayes.MultinomialNB()


def local_train(args={}):
    model = args['model']
    X_train, X_val = split(args['X'], 0.9)
    Y_train, Y_val = split(args['Y'], 0.9)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)
    compute_metrics(Y_val, Y_pred)


if __name__ == '__main__':
    X, Y = data_provider.get_data('../scrapper/out/unijokes.json', 
                                  input_format='hot_vector',
                                  output_format='numerical',
                                  stemmer=nltk.stem.lancaster.LancasterStemmer())
    model = create_model()
    local_train({'X': X, 'Y': Y, 'model': model})
