import argparse
import os
import sys
import time
import warnings
from datetime import datetime

from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from data.dataset import *
from tools.logger import Logger

warnings.simplefilter('ignore')


def main(args):
    print("train beginning!")

    start = time.time()
    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
    dataset = dataset_dic[args.dataset]
    data = dataset(root_path, args.rerange, args.parameter, args.save, args.window, args.overlap)
    print('dataset time')
    print(time.time() - start)

    X = data[:, 1:]
    y = data[:, 0]

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    print("LogisticRegression\n")
    if args.train:
        params_grid = args.grid
    else:
        params_grid = {'C': [1e3], 'random_state': [42]}

    model = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto', random_state=42, max_iter=1e4), params_grid, cv=3, iid=False, scoring="accuracy")
    model.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_predict)
    print('Best score for testing data: %0.2f' % (test_acc * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="linear")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='utd')
    parser.add_argument('-r', '--rerange', type=bool, default=False, choices=[True, False])
    parser.add_argument('-p', '--parameter', nargs='+', default=['laplace', 1e-1])
    parser.add_argument('-s', '--save', type=bool, default=False, choices=[True, False])
    parser.add_argument('-w', '--window', type=int, default=2)
    parser.add_argument('-o', '--overlap', type=bool, default=True, choices=[True, False])
    # train
    parser.add_argument('-t', '--train', type=bool, default=True, choices=[True, False])
    parser.add_argument('-g', '--grid', type=list, default={'C': [1e3]})

    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(root_path, 'logs', stamp)
    os.makedirs(log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_path, 'log_train.txt'))
    print(parser.parse_args())

    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)
    main(parser.parse_args())
    timestats = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(timestats)
