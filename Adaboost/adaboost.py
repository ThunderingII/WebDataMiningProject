import jieba

from sklearn.externals import joblib


class Adaboost(object):
	def __init__(self):
		self.clf = self.load()
		self.algorithm = 'Adaboost'
		self.f1 = 0.990925
		self.className = None
		self.confidence = 1

	def cut(self, msg):
		return ' '.join(jieba.cut(msg))

	def load(self):
		return joblib.load('adaboost.m')

	def predict(self, msg):
		msg_c = self.cut(msg)
		self.className = self.clf.predict([msg])[0]
		return self.className
	

def get_result(msg):
	model = Adaboost()
	model.predict(msg)
	return model.algorithm, model.f1, model.className, model.confidence


if __name__ == '__main__':
	print get_result('您好！我是福州融汇温泉城的高级置业顾问  彭磊，近期我们项目有做些活动，且价位非常优惠，接待点地址：福州市晋安区桂湖。也希望您继续关注')