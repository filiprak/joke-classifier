import logging

import numpy as np
import tflearn

logging.getLogger().setLevel(logging.INFO)

from sklearn import metrics


def is_int(number):
    try:
        int(number)
        return True
    except ValueError:
        return False


def split(array, ratio):
    i = int(len(array) * ratio)
    return array[:i], array[i:]


def pad_sequences(sequences):
    max_len = len(max(sequences, key=lambda s: len(s)))
    for sequence in sequences:
        sequence += [0] * (max_len - len(sequence))
    return sequences


def compute_metrics(Y, Y_pred):
    precision = 100*metrics.precision_score(Y, Y_pred, average="weighted") 
    recall = 100*metrics.recall_score(Y, Y_pred, average="weighted") 
    accuracy = 100*metrics.accuracy_score(Y, Y_pred)

    logging.info("Precision: {:.2f}%".format(precision))
    logging.info("Recall: {:.2f}%".format(recall))
    logging.info("Accuracy: {:.2f}%".format(accuracy))

    return precision, recall, accuracy


def serialize_model(model):
    with model.session.as_default():
        return np.array([np.array(tflearn.variables.get_value(variable).tolist())
                         for variable in model.get_train_vars()])


def update_model(model, data):
  for variable, received in zip(model.get_train_vars(), data):
        model.set_weights(variable, (model.get_weights(variable)+received) / 2.)


def average_models(models):
    return sum(model for model in models) / len(models)
