# -*- coding: utf-8 -*-
# @Date    : 2017/12/12
# @Author  : enzo

import time
import jieba
from sklearn.externals import joblib

class Perceptron(object):
    def __init__(self):
        self.clf = self.loadModel()
        self.algorithm = 'Perceptron'
        self.f1 = 0.98430077
        self.classNum = None
        self.confidence = 1
        self.vectorizer = self.initVectorizer()

    def cut(self, msg):
        return ' '.join(jieba.cut(msg))

    def initVectorizer(self):
        return joblib.load('vectorizer.model')

    def loadModel(self):
        return joblib.load('perceptron.model')

    def predict(self, msg):
        msg_cut = self.cut(msg)
        message = [msg_cut]
        inputVec = self.vectorizer.transform(message)
        self.classNum = self.clf.predict(inputVec)[0]

def get_result(msg):
    model = Perceptron()
    model.predict(msg)
    return model.algorithm, model.f1, model.classNum, model.confidence


if __name__ == '__main__':
    msg = '另为获国家高达xxxx万的补贴'
    t1 = time.time()
    for item in get_result(msg):
        print(item)
    t2 = time.time()
    time_cost = t2 - t1
    print("time cost:{}".format(time_cost))