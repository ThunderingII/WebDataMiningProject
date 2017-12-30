# -*- coding: utf-8 -*-
# @Date    : 2017/12/12
# @Author  : enzo

import time
import jieba
from sklearn.externals import joblib

class Perceptron(object):
    def __init__(self):
        self.clf = self.loadModel()
        self.algorithm = 'naive_bayes'
        self.f1 = 0.98731436
        self.classNum = None
        self.confidence = 1
        self.vectorizer = self.initVectorizer()

    def cut(self, msg):
        return ' '.join(jieba.cut(msg))

    def initVectorizer(self):
        return joblib.load('vectorizer.model')

    def loadModel(self):
        return joblib.load('naive_bayes.model')

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
    msg = '韩依派女装，都市贝贝童装，回馈老顾客，优惠大酬宾，全场半价，欢迎来选购！'
    t1 = time.time()
    for item in get_result(msg):
        print(item)
    t2 = time.time()
    time_cost = t2 - t1
    print("time cost:{}".format(time_cost))