import re
import nltk
import json


class NoStemmer:
    def stem(self, word):
        return word


def get_data(source, representation='bag_of_words', stemmer=NoStemmer()):
    with open(source) as s:
        data = json.load(s)
    print(len(data['jokes']))
    return globals()['get_data_as_' + representation](data, stemmer)


def get_data_as_bag_of_words(data, stemmer):
    return [(to_bag_of_words(joke['joke'], stemmer), 
              to_categorical(joke['categories']))
             for joke in data['jokes']]
    

def to_bag_of_words(joke, stemmer):
    try:
        return [to_bag_of_words.word_indices[stemmer.stem(word.lower())] 
                for word in re.findall(r'\w+', joke)]
    except KeyError as error:
        to_bag_of_words.word_indices[error.args[0].lower()] = to_bag_of_words.index
        to_bag_of_words.index += 1
        return to_bag_of_words(joke, stemmer)

to_bag_of_words.word_indices = {}
to_bag_of_words.index = 0


def to_categorical(categories):
    #TODO, currently it counts categories
    for c in categories:
        to_categorical.t[c.lower()] = ""
    return categories

to_categorical.t = {}


if __name__ == '__main__':
    import sys
    print(get_data(sys.argv[1], stemmer=nltk.stem.snowball.SnowballStemmer('english'))[:10])
    print("Number of unique words after stemming: " + str(to_bag_of_words.index))
    print("Number of categories: " + str(len(to_categorical.t)))
    print("Categories: " + str(to_categorical.t.keys()))
