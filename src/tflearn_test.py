import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

total_counts = Counter()

for idx, row in reviews.iterrows():
    review = row[0]
    for word in review.split(' '):
        total_counts[word] += 1

print("Total words in data set: ", len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])

word2idx = {word: index for index, word in enumerate(vocab)}  ## create the word-to-index dictionary here


def text_to_vector(text):
    vector = np.zeros(len(vocab))
    for word in text.split(' '):
        index = word2idx.get(word, None)
        if index:
            vector[index] += 1
    return vector


word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

Y = (labels == 'positive').astype(np.int_)
records = len(labels)
shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records * test_fraction)], shuffle[int(records * test_fraction):]
trainX, trainY = word_vectors[train_split, :], to_categorical([yv[0] for yv in Y.values[train_split]], 2)
testX, testY = word_vectors[test_split, :], to_categorical([yv[0] for yv in Y.values[test_split]], 2)


# build network
# Network building
def build_model(learning_rate=0.1):
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()

    # Your code #
    net = tflearn.input_data([None, len(vocab)])
    net = tflearn.fully_connected(net, 5, activation='ReLU')
    net = tflearn.fully_connected(net, 5, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=learning_rate, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)

predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)


# Helper function that uses your model to predict sentiment
def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')


while True:
    sentence = input('Test your sentence:')
    test_sentence(sentence)
    if sentence == 'q':
        exit(0)
