import argparse
import os
import sys
import time
import warnings
from datetime import datetime

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

    clf1 = LogisticRegression(penalty='l1', C=1e3, solver='liblinear', multi_class='auto', random_state=42, max_iter=1e4)
    clf2 = SVC(kernel='linear', gamma=1e-4, C=1e4, probability=True, random_state=42)


    for clf, label in zip([clf1, clf2],
                          ['Logistic',
                           'SVM']):
        clf.fit(X_train, y_train)
        scores = accuracy_score(y_test, clf.predict(X_test))
        print("Accuracy: %0.2f "
              % (scores * 100))

    for C in [1e3]:
        lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', multi_class='auto', random_state=42, max_iter=1e4)

        sclf = StackingClassifier(classifiers=[clf1, clf2],
                                  use_probas=True,
                                  average_probas=True,
                                  meta_classifier=lr)
        sclf.fit(X_train, y_train)
        scores = accuracy_score(y_test, sclf.predict(X_test))
        print("Accuracy: %0.2f" % (scores * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="linear")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='utd')
    parser.add_argument('-r', '--rerange', type=bool, default=False, choices=[True, False])
    parser.add_argument('-p', '--parameter', nargs='+', default=['laplace', 1e-1])
    parser.add_argument('-s', '--save', type=bool, default=False, choices=[True, False])
    parser.add_argument('-w', '--window', type=int, default=2)
    parser.add_argument('-o', '--overlap', type=bool, default=True, choices=[True, False])

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