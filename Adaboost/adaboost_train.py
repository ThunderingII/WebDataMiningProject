#coding=utf8

import time
import jieba
import numpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib


train_file = '带标签短信.txt'
test_file = '不带标签短信.txt'

def cut_save():
    with open(train_file) as f, open('cut.txt', 'w') as fw:
        for line in f:
            label, content = line.strip().split('\t')
            wc = ' '.join(jieba.cut(content))
            fw.write((label+'\t'+wc).encode('utf-8')+'\n')

def load():
    X = []
    Y = []
    with open('cut.txt') as f:
        for line in f:
            label, content = line.strip().split('\t')
            X.append(content)
            Y.append(label)
    return X, Y

clf = Pipeline([('vect', TfidfVectorizer()),
				('chi2', SelectKBest(chi2, k=1000)),
                ('clf', AdaBoostClassifier(n_estimators=300)),
])	

cut_save()

X, Y = load()
print len(X), len(Y)

t1 = time.time()
clf = clf.fit(X[:600000], Y[:600000])
t2 = time.time()

def get_report(clf, data, target):
    predicted = clf.predict(data)
    print metrics.classification_report(target, predicted, target_names=['0', '1'])
    print numpy.mean(predicted == target)
    print metrics.confusion_matrix(target, predicted)

get_report(clf, X[:600000], Y[:600000])

t3 = time.time()
get_report(clf, X[600000:], Y[600000:])
t4 = time.time()

print 'Training time: ', t2-t1, ' s'
print 'Test time: ', t4-t3, ' s'

# clf.fit(X, Y)
# joblib.dump(clf, 'adaboost.m')

# from sklearn.cross_validation import cross_val_score, KFold
# from scipy.stats import sem

# def evaluate_cross_validation(clf, X, y, K):
#     # create a k-fold croos validation iterator of k=5 folds
#     cv = KFold(len(y), K, shuffle=True, random_state=0)
#     # by default the score used is the one returned by score method of the estimator (accuracy)
#     scores = cross_val_score(clf, X, y, cv=cv)
#     print scores
#     print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
#         numpy.mean(scores), sem(scores))

# clf = Pipeline([('vect', TfidfVectorizer()),
#                 ('chi2', SelectKBest(chi2, k=1000)),
#                 ('clf', AdaBoostClassifier(n_estimators=300)),
# ])
# evaluate_cross_validation(clf, X, Y, 5)
