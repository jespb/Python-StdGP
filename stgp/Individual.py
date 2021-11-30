from .Node import Node
from .SimpleThresholdClassifier import SimpleThresholdClassifier

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019-2021 J. E. Batista
#

class Individual:
	training_X = None
	training_Y = None

	operators = None
	terminals = None
	max_depth = None

	labelToInt = None
	intToLabel = None

	head = None
	size = 0
	depth = 0

	trainingClassPredictions = None
	trainingValuePredictions = None
	testClassPredictions = None
	testValuePredictions = None
	fitness = None

	model_name = ["SimpleThresholdClassifier"][0]
	model = None

	fitnessType = ["Accuracy", "RMSE"][0]

	def __init__(self, operators, terminals, max_depth):
		self.operators = operators
		self.terminals = terminals
		self.max_depth = max_depth

	def create(self):
		self.head = Node()
		self.head.create(self.operators, self.terminals, self.max_depth, full=True)
		
	def copy(self, head):
		self.head = head


	def __gt__(self, other):
		sf = self.getFitness()
		ss = self.getSize()

		of = other.getFitness()
		os = other.getSize()

		return (sf > of) or (sf == of and ss < os)

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return str(self.head)


	def fit(self, Tr_x, Tr_y):
		'''
		Trains the classifier which will be used in the fitness function
		'''
		if self.model is None:
			self.training_X = Tr_x
			self.training_Y = Tr_y

			self.labelToInt = {}
			self.intToLabel = {}
			classes = list(set(self.training_Y))
			for i in range(len(classes)):
				self.labelToInt[classes[i]] = i
				self.intToLabel[i] = classes[i]

			if self.model_name == "SimpleThresholdClassifier":
				self.model = SimpleThresholdClassifier()

			hyper_X = self.calculate(Tr_x)

			self.model.fit(hyper_X,Tr_y)

			# Binary classification with an extra slot for missing values
			if len(list(set(Tr_y))) <= 3: 
				# If the accuracy is < 0.5, invert the labels
				if self.getAccuracy(Tr_x, Tr_y, pred="Tr") < 0.5:
					self.trainingClassPredictions = None
					self.model.invertPredictions()


	def getSize(self):
		'''
		Returns the total number of nodes within an individual.
		'''
		return self.head.getSize()


	def getDepth(self):
		'''
		Returns the depth of individual.
		'''
		return self.head.getDepth()


	def clone(self):
		'''
		Returns a deep clone of the individual's list of dimensions.
		'''
		ret = Individual()
		ret.copy(head.clone())
		return ret

	def convertLabelsToInt(self, Y):
		ret = [ self.labelToInt[label] for label in Y ]
		return ret

	def convertIntToLabels(self, Y):
		ret = [ self.intToLabel[value] for value in Y ]
		return ret



	def getFitness(self):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness is None:

			if self.fitnessType == "Accuracy":
				self.getTrainingClassPredictions()
				acc = accuracy_score(self.trainingClassPredictions, self.convertLabelsToInt(self.training_Y) )
				self.fitness = acc 

			if self.fitnessType == "RMSE":
				self.getTrainingValuePredictions()
				waf = mean_squared_error(self.trainingValuePredictions, self.convertLabelsToInt(self.training_Y))**0.5
				self.fitness = -waf 

		return self.fitness


	def getTrainingClassPredictions(self):
		if self.trainingClassPredictions is None:
			self.trainingClassPredictions = self.predict(self.training_X, classOutput = True)

		return self.trainingClassPredictions

	def getTestClassPredictions(self,X):
		if self.testClassPredictions is None:
			self.testClassPredictions = self.predict(X, classOutput = True)

		return self.testClassPredictions

	def getTrainingValuePredictions(self):
		if self.trainingValuePredictions is None:
			self.trainingValuePredictions = self.predict(self.training_X, classOutput = False)

		return self.trainingValuePredictions

	def getTestValuePredictions(self,X):
		if self.testValuePredictions is None:
			self.testValuePredictions = self.predict(X, classOutput = False)

		return self.testValuePredictions



	
	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		if pred == "Tr":
			pred = self.getTrainingClassPredictions()
		elif pred == "Te":
			pred = self.getTestClassPredictions(X)
		else:
			pred = self.predict(X)

		return accuracy_score(pred, Y)


	def getRMSE(self, X, Y,pred=None):
		'''
		Returns the individual's WAF.
		'''
		if pred == "Tr":
			pred = self.getTrainingValuePredictions()
		elif pred == "Te":
			pred = self.getTestValuePredictions(X)
		else:
			pred = self.predict(X, classOutput = False)

		return mean_squared_error(pred, Y)**0.5



	def calculate(self, X):
		'''
		Returns the converted input space.
		'''
		return self.head.calculate(X)


	def predict(self, X, classOutput=True):
		'''
		Returns the class prediction of a sample.
		'''
		hyper_X = self.calculate(X)
		if classOutput:
			predictions = self.model.predict(hyper_X)
		else:
			predictions = hyper_X

		return predictions



	def prun(self):
		'''
		Remove the dimensions that degrade the fitness.
		If simp==True, also simplifies each dimension.
		'''
		done = False
		while not done:
			state = str(self.head)
			self.head.prun(self.training_X)
			done = state == str(self.head)



