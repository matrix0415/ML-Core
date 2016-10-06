
def load_data():
	from pandas import DataFrame as df
	import pandas as pd

	training_percentage =0.9

	data = df(pd.read_csv("dataset/machine_data.csv", index_col=False))[:10000]
	version = data[['version']]
	version = [{i: key} for key, i in enumerate(set([i[0] for i in version.values]))]

	version_replace = {}
	[version_replace.update(k) for k in version]
	data['version'] = data['version'].replace(to_replace=version_replace)
	print()
	y = data[['error']].values
	data = data.drop(['date', 'error'], axis=1).values

	training_size =int(training_percentage*len(data))
	testing_size =len(data) -training_size
	return data[:training_size], y[:training_size].reshape(training_size,), data[-testing_size:], y[-testing_size:].reshape(testing_size,)


if __name__ =='__main__':
	from Models.SdA.core import core as SdACore
	from Models.SVM.core import core as SVMCore
	X_train, y_train, X_test, y_test = load_data()

	# SVM Part
	svm_core =SVMCore()
	svm_core.train(X_train, y_train, X_test, y_test)
	#value =svm_core.predict(X_test)


	# SdA Part
	sda_core =SdACore()
	sda_core.train(X_train, y_train, X_test, y_test)
	#value =sda_core.predict(X_test)

	print("SVM Score:", svm_core.model.score)
	print("SDA Score:", sda_core.model.score)

