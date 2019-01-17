import datetime
import json
import pickle
import random
import re

import bcolz
import numpy as np
import tflearn
from nltk.stem import *

from classifiers.network import get_predictions


def splitpercent(first_perc, orig_list):
    a_list = orig_list[:int(len(orig_list) * first_perc)]
    b_list = orig_list[len(a_list):len(orig_list)]
    return a_list, b_list


glove_path = 'glove.6B'

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

print('Loaded glove 50dim model')
print('the', glove['the'])

# prepare weights matrix and input data
with open('wocka.json') as s:
    data = json.load(s)

stemmer = lancaster.LancasterStemmer()
stemmed_jokes = []
stemmed_vocabulary = []
joke_categories = []
assigned_categories = []

max_joke_words = 0

print('Preparing joke dataset...')
jokes_array = []
for i in range(len(data[:20000])):
    if len(data[i]['body']) < 2000:
        jokes_array.append(data[i])

# random.shuffle(jokes_array)
for joke in jokes_array:
    category = joke['category'].strip().lower()
    categories = [category]
    if len(categories) == 0:
        continue
    joke_words = []
    for c in categories:
        if c not in joke_categories:
            joke_categories.append(c)
    for word in re.findall(r'\w+', joke['title'] + ' ' + joke['body']):
        word_s = word.lower()  # stemmer.stem(word)
        if word_s not in stemmed_vocabulary:
            stemmed_vocabulary.append(word_s)
        joke_words.append(word_s)
    max_joke_words = max(max_joke_words, len(joke_words))
    assigned_categories.append(categories)
    stemmed_jokes.append(joke_words)

print('Vocab size = {}, Num jokes = {}, max joke words = {}, num categ = {}'.format(len(stemmed_vocabulary),
                                                                                    len(stemmed_jokes),
                                                                                    max_joke_words,
                                                                                    len(joke_categories)))
empty_word_idx = 0

print('Preparing weight matrix...')
# replace words with glove index
for i in range(len(stemmed_jokes)):
    for j in range(len(stemmed_jokes[i])):
        word = stemmed_jokes[i][j]
        if word in stemmed_vocabulary:
            stemmed_jokes[i][j] = stemmed_vocabulary.index(word) + 1
        else:
            stemmed_jokes[i][j] = empty_word_idx
    # padd with zeros
    for k in range(max_joke_words - len(stemmed_jokes[i])):
        stemmed_jokes[i].append(empty_word_idx)

# prepare weight matrix
matrix_len = len(stemmed_vocabulary) + 1
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

weights_matrix[0] = np.random.normal(scale=0.6, size=(50,))

for i, word in enumerate(stemmed_vocabulary):
    try:
        weights_matrix[i + 1] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i + 1] = np.random.normal(scale=0.6, size=(50,))

print('Words found in glove {}/{}'.format(words_found, len(stemmed_vocabulary)))
print('Weights matrix: ', weights_matrix, weights_matrix.shape)

print('Preparing train and test data...')
# validate / train data to hot vector
for i in range(len(assigned_categories)):
    zero_vec = [0 for e in joke_categories]
    for categ in assigned_categories[i]:
        zero_vec[joke_categories.index(categ)] = 1
    assigned_categories[i] = zero_vec

trainX, testX = splitpercent(0.9, stemmed_jokes)
trainY, testY = splitpercent(0.9, assigned_categories)

print('Creating model...')
# Network building
net = tflearn.input_data([None, max_joke_words])
#create embedding weights, set trainable to False, so weights are not updated
net = tflearn.embedding(net, input_dim=len(stemmed_vocabulary) + 1, output_dim=50, trainable=True,
                        name="EmbeddingLayer")
net = tflearn.flatten(net)
net = tflearn.fully_connected(net, 2048, activation='tanh')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 1024, activation='tanh')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, len(joke_categories), activation='softmax')
net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

# Retrieve embedding layer weights (only a single weight matrix, so index is 0)
embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
# Assign your own weights (for example, a numpy array [input_dim, output_dim])
model.set_weights(embeddingWeights, weights_matrix)

print('Starting learn...')
import utils

date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Train with your custom weights
for i in range(40):
    model.fit(trainX, trainY, n_epoch=5, validation_set=(testX, testY), show_metric=True, batch_size=128,
              run_id='embedding_' + date)
