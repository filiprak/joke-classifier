import re
import nltk
import json
import random


class NoStemmer:
    def stem(self, word):
        return word


def get_data(source, representation='bag_of_words', stemmer=NoStemmer()):
    with open(source) as s:
        data = json.load(s)
    data = globals()['get_data_as_' + representation](data, stemmer)
    #reset_bag_of_words() #uncomment if using get_data more than once
    return data


def get_data_as_bag_of_words(data, stemmer):
    return bag_of_words(data, stemmer, get_word, iter_word)


def get_data_as_ngrams(data, stemmer):
    return bag_of_words(data, stemmer, get_ngram, iter_ngram)


def bag_of_words(data, stemmer, key_getter, key_iter):
    classes = extract_categories(data['jokes'], stemmer)
    return [(to_bag_of_words(joke['joke'], stemmer, key_getter, key_iter),
             to_categorical(joke['categories'], classes, stemmer))
            for joke in data['jokes']
            if joke['categories']]
    

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


def to_bag_of_words(joke, stemmer, key_getter, key_iter):
    try:
        return [to_bag_of_words.word_indices[key_getter(key, stemmer)]
                for key in key_iter(joke)]
    except KeyError as error:
        to_bag_of_words.word_indices[error.args[0]] = to_bag_of_words.index
        to_bag_of_words.index += 1
        return to_bag_of_words(joke, stemmer, key_getter, key_iter)

def reset_bag_of_words():
    to_bag_of_words.word_indices = {}
    to_bag_of_words.index = 1

reset_bag_of_words()


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


def to_hot_vector(joke, nclasses=None):
    if nclasses is None:
        nclasses = to_bag_of_words.index + 1
    out = [0] * nclasses
    for word in joke:
        out[word] = 1
    return out


if __name__ == '__main__':
    import sys
    print(get_data(sys.argv[1],
                   representation='bag_of_words',
                   stemmer=nltk.stem.lancaster.LancasterStemmer())[:10])
    print("Number of unique items after stemming: " + str(to_bag_of_words.index))
