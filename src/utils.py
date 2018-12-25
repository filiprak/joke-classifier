
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
