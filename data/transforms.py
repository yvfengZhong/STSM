import numpy as np


def split(matrix, level, overlap):
    res = []
    for i in range(level):
        length = matrix.shape[0] // 2 ** i
        count = 2 ** i if not overlap else 2 ** (i + 1) - 1
        start = 0
        for j in range(1, count):
            res.append(matrix[start: start + length])
            start += length if not overlap else length // 2

        res.append(matrix[start:])

    return np.array(res)


def stretch(matrix):
    t = []
    for j in range(matrix.shape[0]):
        if j == 0:
            t = np.array(matrix[j][j:])
        else:
            t = np.concatenate((t, matrix[j][j:]))
    return t