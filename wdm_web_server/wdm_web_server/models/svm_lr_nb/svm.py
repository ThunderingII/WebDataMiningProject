import jieba
import math
from sklearn import svm
import scipy.sparse as ss

import os

from sklearn.externals import joblib

division = '> * <'

char_position = {}

size = 0

stop_word = ['0', '1', '  ']

clf = None


def train(filename):
    char_file = open('57M.txt', encoding='utf-8')
    size = 0
    for line in char_file:
        for c in line:
            if c not in char_position:
                char_position[c] = size
                size += 1

    file = open(filename, encoding='utf-8')
    result_file = open('svm-param.txt', mode='w', encoding='utf-8')
    matrix_data = [[], [], []]

    train_y = []
    ln = 0

    for line in file:
        if len(line) < 3:
            continue

        if line[0] == '1':
            train_y.append(1)
        else:
            train_y.append(0)

        # r = jieba.lcut(line)
        r = line
        for i in range(2, len(r)):
            if r[i] in char_position:
                matrix_data[0].append(ln)
                matrix_data[1].append(char_position[r[i]])
                matrix_data[2].append(1)
        ln += 1

    train_x = ss.coo_matrix((matrix_data[2], (matrix_data[0], matrix_data[1])), shape=(ln, size))
    classifier = svm.LinearSVC()

    classifier.fit(train_x, train_y)

    joblib.dump(classifier, 'svm.model')

    result_file.write(str(size))
    result_file.write('\n')
    for t in char_position:
        if t == '\r' or t == '\n':
            continue
        result_file.write(t + division + str(char_position[t]))
        result_file.write('\n')
    result_file.close()
    file.close()


def read_param(filename):
    file = open(filename, encoding='utf-8')
    global size
    i = 0
    for line in file:
        if len(line) < 3:
            continue
        if i == 0:
            size = int(line)
            i += 1
        else:
            data = line.split(division)
            char_position[data[0]] = int(data[1])


def __confidence_compute(data):
    # r = jieba.lcut(data)
    r = data

    global clf
    if clf == None:
        clf = joblib.load('svm.model')
        read_param('svm-param.txt')

    x = [0 for i in range(size)]
    for i in range(2, len(r)):
        term = r[i]
        if term in char_position:
            x[char_position[term]] += 1
    return clf.predict([x])[0]


def get_result(msg):
    print(os.path.abspath('.'))
    print(os.path.realpath(__file__))
    print(os.getcwd())
    print("222222222222222222")
    r = msg
    global clf
    if clf == None:
        clf = joblib.load('svm.model')
        read_param('svm-param.txt')
    x = [0 for i in range(size)]
    for i in range(2, len(r)):
        term = r[i]
        if term in char_position:
            x[char_position[term]] += 1
    className = int(clf.predict([x])[0])
    return 'svm', 0.986959, className, className


def predict(filename):
    file = open(filename, encoding='utf-8')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    output_file = open('test_result.txt', mode='w', encoding='utf-8')

    for line in file:
        if len(line) < 3:
            continue
        confidence = __confidence_compute(line)
        output_file.write(str(confidence) + ' ' + line)
        if line[0] == '1':
            if confidence == 1:
                tp += 1
            else:
                print('1 ' + str(confidence))
                fn += 1
        else:
            if confidence == 0:
                tn += 1
            else:
                print('0 ' + str(confidence))
                fp += 1
    output_file.close()
    print('tp:%d    fp:%d\nfn:%d    tn:%d' % (tp, fp, fn, tn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('recall:%f    precision:%f\nf1:%f' % (recall, precision, recall * precision * 2 / (recall + precision)))


if __name__ == '__main__':
    # train('train.txt')
    # read_param('train-result.txt')
    print(get_result('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注'))
    # predict('test.txt')
