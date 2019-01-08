import re
import nltk
import json
import random

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
        try:
            return [self.word_indices[self.key_getter(key, self.stemmer)]
                    for key in self.key_iter(joke)]
        except KeyError as error:
            self.word_indices[error.args[0]] = self.index
            self.index += 1
            return self.to_bag_of_words(joke)


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
    'step': 100,
    'counter': 0
}


@command()
def get_data(source, input_format='hot_vector', output_format='categorical', ngrams=False):
    if STATE['data'] is None:
        with open(source) as s:
            data = json.load(s)

        STATE['classes'] = extract_categories(data['jokes'], STATE['stemmer'])
        STATE['data'], tokenizer = \
            (get_data_as_ngrams if ngrams else get_data_as_bag_of_words)(data, STATE['stemmer'], STATE['classes'])

    X, Y = zip(*STATE['data'])
    if STATE['X'][input_format] is None:
        if input_format == 'hot_vector':
            STATE['X'][input_format] = [to_hot_vector(x, tokenizer.index) for x in X]
        elif input_format == 'sequential':
            STATE['X'][input_format] = X
        else:
            raise ValueError("Expected values for 'input_format' are 'hot_vector' or 'sequential'")

    if STATE['Y'][output_format] is None:
        if output_format == 'categorical':
            STATE['Y'][output_format] = [to_categorical(y, STATE['classes'], STATE['stemmer']) for y in Y]
        elif output_format == 'numerical':
            STATE['Y'][output_format] = [to_numerical(y, STATE['classes'], STATE['stemmer']) for y in Y]
        else:
            raise ValueError("Expected values for 'output_format' are 'categorical' or 'numerical'")


    X = STATE['X'][input_format]
    Y = STATE['Y'][output_format]
    X = X[counter:counter+['step']]
    return 1


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
    print(get_data(sys.argv[1],
                   representation='bag_of_words',
                   stemmer=nltk.stem.lancaster.LancasterStemmer())[:10])
