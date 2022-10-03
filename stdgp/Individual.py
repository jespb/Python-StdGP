from .Node import Node
from .SimpleThresholdClassifier import SimpleThresholdClassifier

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error



# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-StdGP
#
# Copyright ©2019-2022 J. E. Batista
#

class Individual:
	training_X = None
	training_Y = None

	operators = None
	terminals = None
	max_depth = None

	head = None
	size = 0
	depth = 0

	trainingPredictions = None
	testPredictions = None
	fitness = None

	model = None

	def __init__(self, operators, terminals, max_depth, model_name="SimpleThresholdClassifier", fitnessType="Accuracy"):
		self.operators = operators
		self.terminals = terminals
		self.max_depth = max_depth
		self.model_name = model_name
		self.fitnessType = fitnessType

	def create(self,rng):
		self.head = Node()
		self.head.create(rng, self.operators, self.terminals, self.max_depth, full=True)
		
	def copy(self, head):
		self.head = head



	def __gt__(self, other):
		sf = self.getFitness()
		ss = self.getSize()

		of = other.getFitness()
		os = other.getSize()

		return (sf > of) or \
				(sf == of and ss < os)

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return str(self.head)


	def createModel(self):
		if self.model_name == "SimpleThresholdClassifier":
			return SimpleThresholdClassifier()


	def fit(self, Tr_x, Tr_y):
		'''
		Trains the classifier which will be used in the fitness function
		'''
		if self.model is None:
			self.training_X = Tr_x
			self.training_Y = Tr_y

			self.model = self.createModel()
	
			hyper_X = self.convert(Tr_x)

			self.model.fit(hyper_X,Tr_y)

	def getHead(self):
		return self.head.clone()


	def getSize(self):
		'''
		Returns the total number of nodes within an individual.
		'''
		if not self.size:
			self.size = self.head.getSize() 
		return self.size

	def getDepth(self):
		'''
		Returns the depth of individual.
		'''
		if not self.depth:
			self.depth = self.head.getDepth() 
		return self.depth 






	def getFitness(self, tr_x = None, tr_y = None):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness is None:
			if not tr_x is None:
				self.training_X = tr_x
			if not tr_y is None:
				self.training_Y = tr_y


			if self.fitnessType == "Accuracy":
				self.fit(self.training_X, self.training_Y)
				self.getTrainingPredictions()
				acc = accuracy_score(self.trainingPredictions, self.training_Y)
				self.fitness = acc 

			if self.fitnessType == "MSE":
				self.fit(self.training_X, self.training_Y)
				self.getTrainingPredictions()
				mse = -1 * mean_squared_error(self.trainingPredictions, self.training_Y)
				self.fitness = mse 

			if self.fitnessType == "WAF":
				self.fit(self.training_X, self.training_Y)
				self.getTrainingPredictions()
				waf = f1_score(self.trainingPredictions, self.training_Y, average="weighted")
				self.fitness = waf 

			if self.fitnessType == "2FOLD":
				hyper_X = self.convert(self.training_X)

				X1 = hyper_X.iloc[:len(hyper_X)//2]
				Y1 = self.training_Y[:len(self.training_Y)//2]
				X2 = hyper_X.iloc[len(hyper_X)//2:]
				Y2 = self.training_Y[len(self.training_Y)//2:]

				M1 = self.createModel()
				M1.fit(X1,Y1)
				P1 = M1.predict(X2)

				M2 = self.createModel()
				M2.fit(X2,Y2)
				P2 = M2.predict(X1)

				f1 = accuracy_score(P1, Y2)
				f2 = accuracy_score(P2, Y1)
				self.fitness = (f1+f2)/2

		return self.fitness


	def getTrainingMeasure(self):
		if self.fitnessType in ["Accuracy", "2FOLD"]:
			self.getTrainingPredictions()
			return accuracy_score(self.trainingPredictions, self.training_Y)
			
		if self.fitnessType == "MSE":
			self.getTrainingPredictions()
			return -1 * mean_squared_error(self.trainingPredictions, self.training_Y)

		if self.fitnessType == "WAF":
			self.getTrainingPredictions()
			return f1_score(self.trainingPredictions, self.training_Y, average="weighted")


	def getTestMeasure(self, test_X, test_Y):
		if self.fitnessType in ["Accuracy", "2FOLD"]:
			self.getTestPredictions(test_X)
			return accuracy_score(self.testPredictions, test_Y)
			
		if self.fitnessType == "MSE":
			self.getTestPredictions(test_X)
			return -1 * mean_squared_error(self.testPredictions, test_Y)

		if self.fitnessType == "WAF":
			self.getTestPredictions(test_X)
			return f1_score(self.testPredictions, test_Y, average="weighted")



	def getTrainingPredictions(self):
		if self.trainingPredictions is None:
			self.trainingPredictions = self.predict(self.training_X)

		return self.trainingPredictions

	def getTestPredictions(self, X):
		if self.testPredictions is None:
			self.testPredictions = self.predict(X)

		return self.testPredictions


	
	def getMSE(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return -1 * mean_squared_error(pred, Y)

	
	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return accuracy_score(pred, Y)


	def getWaF(self, X, Y,pred=None):
		'''
		Returns the individual's WAF.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return f1_score(pred, Y, average="weighted")


	def getKappa(self, X, Y,pred=None):
		'''
		Returns the individual's kappa value.
		'''
		if pred == "Tr":
			pred = self.getTrainingPredictions()
		elif pred == "Te":
			pred = self.getTestPredictions(X)
		else:
			pred = self.predict(X)

		return cohen_kappa_score(pred, Y)



	def calculate(self, X):
		'''
		Return the position of a sample in the output space.
		'''
		return self.head.calculate(X)


	def convert(self, X):
		'''
		Returns the converted input space.
		'''
		ret = pd.DataFrame()
		a = self.head.calculate(X)
		ret["#0"] = a
		return ret


	def predict(self, X):
		'''
		Returns the class prediction of a sample.
		'''
		hyper_X = self.convert(X)
		predictions = self.model.predict(hyper_X)

		return predictions



	def prun(self):
		done = False
		while not done:
			d = self.head
			state = str(d)
			d.prun(self.training_X)
			done = state == str(d)



