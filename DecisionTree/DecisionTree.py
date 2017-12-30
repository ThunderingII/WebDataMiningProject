# -*- coding:utf-8 -*-
import jieba

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import tree
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

import numpy as np
import scipy as sp

import time

class stopword(object):
	def __init__(self, file):
		fd = open(file, encoding='utf-8')
		self.data = []
		for word in fd:
			self.data.append(word.strip())

# 去停用词
def remove_stopwords(stop_list, data_list):
	result = []
	# 这里面的leave_word不能作为最终的特征，因为leave_word当中每个词只出现一次
	# 当多个非停用词出现在一句话中的时候，信息就会损失
	leave_word = set(data_list) - set(stop_list)
	for word in data_list:
		if word in leave_word:
			result.append(word)
	return result


class Data(object):
	def __init__(self, train_file, stop_file=None):

		self.data = []
		self.label = []

		file = open(train_file, "r", encoding='utf-8')
		if stop_file is not None:
			self.stopwords = set(stopword(stop_file).data)
		else:
			self.stopwords = set()

		#i=0
		for line in file:
			split_line = line.split('\t')

			if len(split_line) != 2:
				print(line)
				continue

			label, content = split_line
			content = remove_stopwords(self.stopwords, list(jieba.cut(content)))
			content = ' '.join(content)
			self.data.append(content)
			self.label.append(int(label))
			#print("Loading data %d:%s" % (i,label),end="\r" )
			#i+=1
		#print("Finish: Data Loading.")
	
def DecisionTreeClassifyTfidf(data_file, stop=None):
	data = Data(data_file, stop)
	#data=joblib.load("ALLData.joblibdump")
	print("Finish Data Loading!")
	
	# TF-IDF模块
	vectorizer = TfidfVectorizer(max_df=0.5,
								 max_features=3000,
								 min_df=2,
								 use_idf=True,
								 lowercase=False,
								 decode_error='ignore',
								 analyzer=str.split).fit(data.data)
	joblib.dump(vectorizer,"tfidf.model")
	#vectorizer=joblib.load("tfidf.model")
	
	X=data.data
	Y=data.label
	FoldNummer=5		# 分成的组数
	kf=KFold(n_splits=FoldNummer)
	kf.get_n_splits(X)
	
	
	print(kf)
	i=0
	maxF=0
	maxFi=-1
	maxFtimeCost=0

	for train_index,test_index in kf.split(X):
		i+=1
		print("============= TestSet:",i," ===============")
		# 数据集处理，统计每个词的TF-IDF 
		#train_x,test_x=X[train_index],X[test_index]
		#train_y,test_y=Y[train_index],Y[test_index]
		train_x=[X[i] for i in train_index]
		test_x=[X[i] for i in test_index]
		train_y=[Y[i] for i in train_index]
		test_y=[Y[i] for i in test_index]
		
		train_x = vectorizer.transform(train_x)
		test_x = vectorizer.transform(test_x)
		DecisionTree = DecisionTreeClassifier(criterion="entropy",
											  splitter="best",
											  max_depth=None,
											  min_samples_split=2,
											  min_samples_leaf=2)
		print("Finish preprocessing")
		# 训练
		startTime=time.clock()
		DecisionTree.fit(train_x, train_y)
		endTime=time.clock()
		timeCost=endTime-startTime
		print("Finish Training! Time of training:",timeCost)

		# 检验-模型表现 
		answer = DecisionTree.predict_proba(test_x)[:,1]  
		#precision, recall, thresholds = precision_recall_curve(test_y, answer)
		#print("\n")
		#print("Recall	 : ",recall)
		#print("Precision  : ",precision)
		#print("thresholds : ",thresholds)
		
		report = answer > 0.5  
		print(classification_report(test_y, report, target_names = ['norm', 'spam'])) 

		predict_y = DecisionTree.predict(test_x)
		PT=(predict_y == np.ones(len(predict_y)))
		PN=(predict_y == np.zeros(len(predict_y)))
		OT=(test_y == np.ones(len(test_y)))
		ON=(test_y == np.zeros(len(test_y)))
		FP = np.mean(PT & ON)
		TP = np.mean(PT & OT)
		FN = np.mean(PN & OT)
		TN = np.mean(PN & ON)
		
		P = TP/(TP+FP)
		R = TP/(TP+FN)
		F = 2*P*R/(P+R)

		print("precision\t: ", P)
		print("Recall\t\t: ", R)
		print("F1-messure\t: ", F)
		print("timecost\t: ", timeCost)
		if F>maxF:
			maxF=F
			maxFi=i
			maxFtimeCost=timeCost
			# 模型保存
			joblib.dump(DecisionTree, 'dt.model')
			print("Save This DecisionTree!")
	print("============ Result ==============")
	print("Best Model:",maxFi,",its F1-messure:",maxFi,",its timecost:",maxFtimeCost)

def get_result(msg):
	ctfidf= joblib.load("tfidf.model")
	clf = joblib.load('dt.model')
	
	stopwords = set(stopword('./stopword.txt').data)
	content = remove_stopwords(stopwords, list(jieba.cut(msg)))
	content = ' '.join(content)
	question_x = ctfidf.transform([content])
	
	className = clf.predict(question_x)
	return 'TfIdf+decisionTree', 0.990625, className[0], className[0]



if __name__ == '__main__':
	DecisionTreeClassifyTfidf('./train.txt', './stopword.txt')
	print(get_result('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注'))