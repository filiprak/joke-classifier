import re
import nltk
import json
import random
import logging
import numpy as np

from pulsar.async.proxy import command

import utils


class NoStemmer:
    def stem(self, word):
        return word


class Tokenizer:
    def __init__(self, stemmer, key_getter, key_iter):
        self.word_indices = {}
        self.index = 1
        self.stemmer = stemmer
        self.key_getter = key_getter
        self.key_iter = key_iter

    def to_bag_of_words(self, joke):
        result = []
        for key in self.key_iter(joke):
            try:
                idx = self.word_indices[self.key_getter(key, self.stemmer)]
                result.append(idx)

            except KeyError as error:
                self.word_indices[error.args[0]] = self.index
                result.append(self.index)
                self.index += 1

        return result


def initial_state():
    return {
        'X': {
            'sequential': None,
            'hot_vector': None
        },
        'Y': {
            'categorical': None,
            'numerical': None,
        },

        'model_params': {
            'input_length': {
                'sequential': None,
                'hot_vector': None
            },
            'output_length': {
                'categorical': None,
                'numerical': None,
            }
        },
        'data': None,
        'classes': None,
        'stemmer': nltk.stem.lancaster.LancasterStemmer(),
        'step': 100,
        'counter': 0,

        'source': '../scrapper/out/unijokes.json',
        'max_jokes': 6000,
    }


STATE = initial_state()


@command()
def data_provider_info(request):
    return {
        'X_length': {
            'sequential': len(STATE['X']['sequential']),
            'hot_vector': len(STATE['X']['hot_vector'])
        },
        'Y_length': {
            'categorical': len(STATE['Y']['categorical']),
            'numerical': len(STATE['Y']['numerical']),
        },
        'model_params': STATE['model_params'],
        'classes_num': len(STATE['classes']),
        'step': STATE['step'],
        'counter': STATE['counter'],
        'source': STATE['source'],
    }


@command()
def init_data_provider_command(request, ngrams=False):
    init_data_provider(ngrams)
    return 'ok'


@command()
def get_data_command(request, args={}):
    if 'input_format' not in args:
        raise ValueError("get_data_command(): Expected values for 'input_format' are 'hot_vector' or 'sequential'")
    if 'output_format' not in args:
        raise ValueError("get_data_command(): Expected values for 'output_format' are 'categorical' or 'numerical'")

    ngrams = False
    if 'ngrams' in args:
        ngrams = args['ngrams']

    return get_data(args['input_format'], args['output_format'], ngrams, False)


def init_data_provider(ngrams=False):
    logging.info('Data provider, initializing: ngrams = {}'.format(ngrams))
    logging.info('Data provider, loading file: ' + STATE['source'])
    with open(STATE['source']) as s:
        data = json.load(s)

    data['jokes'] = data['jokes'][:STATE['max_jokes']]
    logging.info('Data provider, extracting categories...')
    STATE['classes'] = extract_categories(data['jokes'], STATE['stemmer'])
    logging.info('Data provider, tokenizing data...')
    STATE['data'], STATE['tokenizer'] = \
        (get_data_as_ngrams if ngrams else get_data_as_bag_of_words)(data, STATE['stemmer'], STATE['classes'])

    X, Y = zip(*STATE['data'])
    STATE['X']['hot_vector'] = np.empty((len(X), STATE['tokenizer'].index))
    for i, e in enumerate(X):
        STATE['X']['hot_vector'][i] = to_hot_vector(e, STATE['tokenizer'].index)
    STATE['X']['sequential'] = utils.pad_sequences(X)
    STATE['Y']['categorical'] = np.array([to_categorical(y, STATE['classes'], STATE['stemmer']) for y in Y])
    STATE['Y']['numerical'] = np.array([to_numerical(y, STATE['classes'], STATE['stemmer']) for y in Y])

    STATE['model_params']['input_length']['hot_vector'] = len(STATE['X']['hot_vector'][0])
    STATE['model_params']['input_length']['sequential'] = len(STATE['X']['sequential'][0])
    STATE['model_params']['output_length']['categorical'] = len(STATE['Y']['categorical'][0])
    STATE['model_params']['output_length']['numerical'] = 1

    logging.info(
        'Data provider, finished loading [' + str(len(data['jokes'])) + ' jokes] from file: ' + STATE['source'])


def get_data(input_format='hot_vector', output_format='categorical', ngrams=False, all_data=False):
    if input_format not in ['hot_vector', 'sequential']:
        raise ValueError("Expected values for 'input_format' are 'hot_vector' or 'sequential'")
    if output_format not in ['numerical', 'categorical']:
        raise ValueError("Expected values for 'output_format' are 'categorical' or 'numerical'")

    X = STATE['X'][input_format]
    Y = STATE['Y'][output_format]
    if not all_data:
        X = X[STATE['counter']:STATE['counter'] + STATE['step']]
        Y = Y[STATE['counter']:STATE['counter'] + STATE['step']]
        STATE['counter'] = STATE['counter'] + STATE['step']
        logging.info("Get data: {}".format(STATE['counter']))
    return X, Y


def get_data_as_bag_of_words(data, stemmer, classes):
    return np.array(bag_of_words(data, stemmer, classes, get_word, iter_word))


def get_data_as_ngrams(data, stemmer, classes):
    return np.array(bag_of_words(data, stemmer, classes, get_ngram, iter_ngram))


def bag_of_words(data, stemmer, classes, key_getter, key_iter):
    tokenizer = Tokenizer(stemmer, key_getter, key_iter)
    return ([(tokenizer.to_bag_of_words(joke['joke']),
              joke['categories'])
             for joke in data['jokes']
             if joke['categories']],
            tokenizer)


def extract_categories(jokes, stemmer):
    output = {}
    index = 1
    for joke in jokes:
        for category in joke['categories']:
            stemmed = stemmer.stem(category.lower())
            if stemmed not in output:
                output[stemmed] = index
                index += 1
    return output


def get_ngram(ngram, stemmer):
    return " ".join(get_word(word, stemmer) for word in ngram)


def get_word(word, stemmer):
    return stemmer.stem(word.lower())


def iter_ngram(jokes, grams=2):
    return nltk.ngrams(iter_word(jokes), grams)


def iter_word(joke):
    return re.findall(r'\w+', joke)


def to_categorical(categories, classes, stemmer):
    out = [0] * (len(classes) + 1)
    out[classes[stemmer.stem(random.choice(categories).lower())]] = 1
    return np.array(out)


def to_numerical(categories, classes, stemmer):
    return classes[stemmer.stem(random.choice(categories).lower())]


def to_hot_vector(joke, nclasses):
    out = [0] * nclasses
    for word in joke:
        out[word] = 1
    return np.array(out)


if __name__ == '__main__':
    import sys

    STATE['source'] = sys.argv[1]
    init_data_provider()
    print(get_data()[:2])
