import logging

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
