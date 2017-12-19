import jieba
import json

from sklearn.externals import joblib


class Adaboost(object):
	def __init__(self):
		self.clf = self.load()
		self.algorithm = 'Adaboost'
		self.f1 = 0.99
		self.classNum = None
		self.confidence = 1

	def cut(self, msg):
		return ' '.join(jieba.cut(msg))

	def load(self):
		return joblib.load('adaboost.m')

	def get_result(self, msg):
		msg_c = self.cut(msg)
		self.classNum = self.clf.predict([msg])
		data = {
					'algorithm'	: 	self.algorithm,
			   		'f1'		: 	self.f1,
			   		'classNum'	: 	self.classNum[0],
			   		'confidence': 	self.confidence
			   	}
		json_data = json.dumps(data)
		return json_data
	

if __name__ == '__main__':
	model = Adaboost()
	json_data = model.get_result('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注')
	print json_data