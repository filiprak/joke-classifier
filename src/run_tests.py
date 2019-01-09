import sys

from nltk.stem import *

import data_provider
import classifiers.network
import classifiers.svm
import classifiers.bayes

STEMMERS = [
    data_provider.NoStemmer(),
    lancaster.LancasterStemmer(),
    porter.PorterStemmer(),
    snowball.SnowballStemmer(language='english')
]

TEXT_REPRESENTATIONS = [
    'bag-of-words',
    'ngrams'
]

MODELS = [
    classifiers.network.dnn_model,
    classifiers.network.lstm_model,
]

C_VALUES = [
    1e3,
    1e2,
    1e4,
    1e1,
    1e5
]

ITER_VALUES = [
    1000,
    10000,
    20000,
    30000,
    40000,
    50000,
    100000
]


def main():
    for stemmer in STEMMERS:
        for text_representation in TEXT_REPRESENTATIONS:
            for model in MODELS:
                classifiers.network.local_train(stemmer, text_representation, model)
            for C in C_VALUES:
                for iter_value in ITER_VALUES:
                    classifiers.svm.local_train(stemmer, text_representation, C, iter_value)
            classifiers.bayes.local_train(stemmer, text_representation)


if __name__ == '__main__':
    sys.exit(main())
