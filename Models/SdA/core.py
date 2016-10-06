# Romeu, P., Zamora-Martínez, F., Botella-Rocamora, P., & Pardo, J. (2015).
# Stacked Denoising Auto-Encoders for Short-Term Time Series Forecasting.
# In Artificial Neural Networks (pp. 463-486). Springer International Publishing.

from Models import modelObj


class core(modelObj):
	class config:
		batch = None
		epoch = 20
		hiddenLayers = [1536, 1024, 512]
		noiseLayers = [0.6, 0.4, 0.2]
		learningMethod = 'classification'  # ["classification",           "regression"]
		lossMode = 'categorical_crossentropy'  # ['categorical_crossentropy', 'mape']


	def __prepare(self, X_train, y_train):
		from numpy import copy

		Xshape, yshape =X_train.shape, y_train.shape
		self.config.hiddenLayers.insert(0,Xshape[1])
		print(yshape)
		if self.config.batch == None: self.config.batch = 0.3 * X_train.shape[0]
		if len(yshape)==1: self.output_dim =1
		elif len(yshape)==2: self.output_dim =yshape[1]
		self.method =self.config.learningMethod.lower()

		return copy(X_train)


	def setConfig(self, **kwargs):
		if 'batch' in kwargs: self.config.batch =kwargs['batch']
		if 'epoch' in kwargs: self.config.batch =kwargs['epoch']
		if 'hiddenLayers' in kwargs: self.config.batch =kwargs['hiddenLayers']
		if 'noiseLayers' in kwargs: self.config.batch =kwargs['noiseLayers']
		if 'learningMethod' in kwargs: self.config.batch =kwargs['learningMethod']
		if 'lossMode' in kwargs: self.config.batch =kwargs['lossMode']


	def train(self, X_train, y_train, X_test, y_test):
		from keras.models import Sequential
		from keras.layers.noise import GaussianNoise
		from keras.layers.core import Dense, AutoEncoder, Dropout

		X_train_tmp =self.__prepare(X_train, y_train)
		epoch = self.config.epoch
		batch = self.config.batch
		nb_noise_layers = self.config.noiseLayers
		nb_hidden_layers = self.config.hiddenLayers
		lossMode =self.config.lossMode
		output_dim =self.output_dim
		trained_encoders = []

		if len(nb_noise_layers) !=len(nb_hidden_layers)-1:
			raise Exception('Noise Layers Error', 'Noise layer length is not correct. Hidden:%s Noise:%s'%(
					len(nb_hidden_layers), len(nb_noise_layers)
			))

		else:
			for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
				print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
				encoder =Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid')
				decoder =Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')

				ae = Sequential()
				ae.add(GaussianNoise(nb_noise_layers[i - 1], input_shape=(n_in,)))
				ae.add(Dropout(nb_noise_layers[i - 1], input_shape=(n_in,)))
				ae.add(AutoEncoder(encoder=encoder, decoder=decoder))
				ae.compile(loss='mape', optimizer='rmsprop')
				ae.fit(X_train_tmp, X_train_tmp, batch_size=batch, nb_epoch=epoch)
				trained_encoders.append(ae.layers[2].encoder)

				ae.layers[2].output_reconstruction = False
				ae.compile(loss='mape', optimizer='rmsprop')
				X_train_tmp = ae.predict(X_train_tmp)

			print("Fine Tuning")
			model = Sequential()

			for encoder in trained_encoders:
				model.add(encoder)

			model.add(Dense(input_dim=nb_hidden_layers[-1], output_dim=output_dim, activation='linear'))

			if self.method =="regression":
				model.compile(loss=lossMode, optimizer='rmsprop')
				model.fit(X_train, y_train, batch_size=batch, nb_epoch=epoch, validation_data=(X_test, y_test))

			elif self.method =="classification":
				model.compile(loss=lossMode, optimizer='rmsprop')
				model.fit(X_train, y_train, batch_size=batch, nb_epoch=epoch, show_accuracy=True, validation_data=(X_test, y_test))

			self.model.name = "sda"
			self.model.model = model
			self.model.score = model.evaluate(X_test, y_test)
			print("Score: ", self.model.score)

			return self.model


	def predict(self, X):
		return self.model.model.predict(X)


	def evaluate(self, X, y):
		return self.model.model.evaluate(X, y)


	def loadModel(self):
		from keras.models import model_from_json

		if self.model.folderPath == None:
			print("Fill up the Models.ModelObj.GlobalModel.folderPath first.")
		else:
			model = model_from_json(open("%s/architecture.json" % self.model.folderPath).read())
			model.load_weights("%s/weights.h5" % self.model.folderPath)
			self.model.name = "sda"
			self.model.model = model
			print("Load Model: ", self.model.name)


	def saveModel(self):
		if self.model.folderPath == None:
			print("Fill up the Models.ModelObj.GlobalModel.folderPath first.")
		else:
			open("%s/architecture.json" % (self.model.folderPath), 'w').write(self.model.model.to_json())
			self.model.model.save_weights('%s/weights.h5' % (self.model.folderPath))