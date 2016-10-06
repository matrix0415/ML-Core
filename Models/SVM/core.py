from Models import modelObj
from sklearn import svm

class core(modelObj):
	class config:
		kernel = 'linear'
		verbose = True
		learningMethod = 'classification'  # ["classification",           "regression"]


	def train(self, X_train, y_train, X_test, y_test):
		model =None
		kernel =self.config.kernel
		verbose =self.config.verbose

		if self.config.learningMethod =="classification": model =svm.SVC(kernel=kernel, verbose=verbose)
		elif self.config.learningMethod =="regression": model =svm.SVR(kernel=kernel, verbose=verbose)
		self.model.model =model.fit(X_train, y_train)
		return self.model


	def predict(self, X_test):
		return self.model.model.predict(X_test)


	def evaluate(self, X, y):
		self.model.score =self.model.model.score(X, y)
		return self.model.score


	def saveModel(self):
		pass


	def loadModel(self):
		pass



if __name__ == "__main__":
	from pandas import DataFrame as df
	import pandas as pd

	data =df(pd.read_csv("dataset/machine_data.csv", index_col=False))