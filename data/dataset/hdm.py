import glob
import os
import time

import pandas as pd

from data.spd import kernelTrans as ker
from data.transforms import *
from tools.metrics import AverageMeter


def hdm(root_path, rerange, para, save, window, overlap):
    filename = root_path + "data/dataset/hdm/*" + ".txt"
    filename = glob.glob(filename)
    name = {}
    k = 0

    avg = AverageMeter()
    j = 0
    for i in range(len(filename)):
        temp = filename[i].split('_')
        if temp[-3] not in name:
            name[temp[-3]] = k
            k += 1

    flag2 = True
    x = []
    for action in name:
        filename = root_path + "data/dataset/hdm/*" + action + "*.txt"
        filename = glob.glob(filename)
        flag1 = True
        X = []
        for f in filename:
            data = pd.read_csv(f, header=None, sep='\s+')
            temp = data.iloc[:, :].values
            avg.update(temp.shape[0])
            if temp.shape[0] < 16:
                j += 1

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

            a = np.array([name[action]]).reshape(1, -1)
            b = t.reshape(1, -1)
            temp = np.concatenate((a, b), axis=1)
            if flag1:
                flag1 = False
                X = temp
            else:
                X = np.append(X, temp, 0)

        if flag2:
            x = X
            flag2 = False
        else:
            x = np.append(x, X, 0)

    print("frames min:", avg.min)
    print("frames max:", avg.max)
    print("frames avg:", avg.avg)

    if save:
        data_path = os.path.join(root_path, "hdm_" + para[0] + "_" + str(para[1]) + ".txt")
        np.savetxt(data_path, x, fmt='%1.6f', delimiter=' ')

    return x


if __name__ == "__main__":
    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)
    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
    res = hdm(root_path, False, ['laplace', '5e-4'], False, 4, True)
    print(len(res))
    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)