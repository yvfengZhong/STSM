import os
import time

import pandas as pd

from data.transforms import *
from data.spd import kernelTrans as ker
from tools.metrics import AverageMeter


def g3d(root_path, rerange, para, save, window, overlap):
    x = []
    flag = True
    avg = AverageMeter()

    filename = []
    for line in open(os.path.join(root_path, 'data/dataset/g3d_path.txt')):
        filename.append(os.path.join(root_path, line.split('\n')[0]))

    for f in filename:
        data = pd.read_csv(f, header=None, sep=',')
        temp = data.iloc[:, :].values
        avg.update(temp.shape[0])

        if rerange:
            for i in range(len(temp)):
                temp[i] = np.concatenate((temp[i][0::3], temp[i][1::3], temp[i][2::3]))

        t = []
        matrixes = split(temp, window, overlap)
        for matrix in matrixes:
            spd = ker(matrix.T, para)
            vec = stretch(spd)
            t.append(vec)

        t = np.array(t)

        a = np.array(int(f.split('/')[-1].split('_')[0][-1])).reshape(1, -1)
        b = t.reshape(1, -1)
        temp = np.concatenate((a, b), axis=1)

        if flag:
            x = temp
            flag = False
        else:
            x = np.append(x, temp, 0)

    print("frames min:", avg.min)
    print("frames max:", avg.max)
    print("frames avg:", avg.avg)

    if save:
        data_path = os.path.join(root_path, "g3d_" + para[0] + "_" + str(para[1]) + ".txt")
        np.savetxt(data_path, x, fmt='%1.6f', delimiter=' ')

    return x


if __name__ == "__main__":
    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)
    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
    res = g3d(root_path, False, ['laplace', '5e-1'], False, 5, True)
    print(len(res))
    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)