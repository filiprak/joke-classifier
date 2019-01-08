import re
import nltk
import json
import random
import logging

from pulsar.async.proxy import command


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


STATE = {
    'X': {
        'sequential': None,
        'hot_vector': None
    },
    'Y': {
        'categorical': None,
        'numerical': None,
    },
    'data': None,
    'classes': None,
    'stemmer': nltk.stem.lancaster.LancasterStemmer(),
    'step': 1000,
    'counter': 0,

    'source': '../scrapper/out/unijokes.json',
    'max_jokes': 100000,
}


@command()
def data_provider_info():
    return {
        'X': STATE['X'],
        'Y': STATE['Y'],
        'step': STATE['step'],
        'counter': STATE['counter'],
        'source': STATE['source'],
    }


@command()
def init_data_provider_command(ngrams=False):
    init_data_provider(ngrams)
    return 'ok'


@command()
def get_data_command(input_format='hot_vector', output_format='categorical', ngrams=False):
    return get_data(input_format, output_format, ngrams, all_data=False)


def init_data_provider(ngrams=False):
    logging.info('Data provider, loading file: ' + STATE['source'])
    with open(STATE['source']) as s:
        data = json.load(s)

    data['jokes'] = data['jokes'][:STATE['max_jokes']]

    logging.info('Data provider, extracting categories...')

    STATE['classes'] = extract_categories(data['jokes'], STATE['stemmer'])
    logging.info('Data provider, tokenizing data...')
    STATE['data'], tokenizer = \
        (get_data_as_ngrams if ngrams else get_data_as_bag_of_words)(data, STATE['stemmer'], STATE['classes'])

    X, Y = zip(*STATE['data'])

    STATE['X']['hot_vector'] = [to_hot_vector(x, tokenizer.index) for x in X]
    STATE['X']['sequential'] = X

    STATE['Y']['categorical'] = [to_categorical(y, STATE['classes'], STATE['stemmer']) for y in Y]
    STATE['Y']['numerical'] = [to_numerical(y, STATE['classes'], STATE['stemmer']) for y in Y]

    logging.info('Data provider, loading data finished, file: ' + STATE['source'])


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
    return bag_of_words(data, stemmer, classes, get_word, iter_word)


def get_data_as_ngrams(data, stemmer, classes):
    return bag_of_words(data, stemmer, classes, get_ngram, iter_ngram)


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
    return out


def to_numerical(categories, classes, stemmer):
    return classes[stemmer.stem(random.choice(categories).lower())]


def to_hot_vector(joke, nclasses):
    out = [0] * nclasses
    for word in joke:
        out[word] = 1
    return out


if __name__ == '__main__':
    import sys
    STATE['source'] = sys.argv[1]
    init_data_provider()
    print(get_data()[:2])
