import numpy as np


def foreach(fn, collection):
    for item in collection:
        fn(item)


def read_padded_csv(path, pad):
    max_length = 0
    lines = []

    with open(path, 'r+') as f:
        for line in f:
            c = list(map(lambda t: float(t), line.split(',')))
            len_c = len(c)
            if len_c > max_length:
                max_length = len_c

            lines.append(c)

    for line in lines:
        while len(line) < max_length:
            line.append(pad)

    return np.array(lines)
