from .Node import Node
from .Constants import * 

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

class Individual:
	head = None

	trainingAccuracy = None
	testAccuracy = None

	trainingRMSE = None
	testRMSE = None

	trainingPredictions = None
	testPredictions = None
	trainingClassPredictions = None
	testClassPredictions = None


	def __init__(self, node = None, fromString = None):
		if fromString == None:
			self.head = Node() if node == None else node
		else:
			self.head = Node(fromString = fromString.split())

	def predict(self, sample):
		return 0 if self.calculate(sample) < 0.5 else 1

	def calculate(self, sample):
		return self.head.calculate(sample)

	def getHead(self):
		return self.head.clone()

	def getDepth(self):
		return self.head.getDepth()

	def getSize(self):
		return self.head.getSize()

	def __str__(self):
		return str(self.head)

	def __gt__(self, other):
		# Using RMSE as fitness
		if OUTPUT == "Classification":
			return self.getTrainingAccuracy() > other.getTrainingAccuracy()
		else:
			return self.getTrainingRMSE() < other.getTrainingRMSE()



	## FITNESS

	def getFitness(self):
		if self.fitness == None:
			self.fitness = self.getTrainingRMSE()
		return self.fitness


	def getTrainingPredictions(self):
		if self.trainingPredictions == None:
			self.trainingPredictions = [ self.calculate(sample) for sample in getTrainingSet() ]
		return self.trainingPredictions

	def getTrainingClassPredictions(self):
		if self.trainingClassPredictions == None:
			self.trainingClassPredictions = [ 0 if v < 0.5 else 1 for v in self.getTrainingPredictions() ]
		return self.trainingClassPredictions

	def getTestPredictions(self):
		if self.testPredictions == None:
			self.testPredictions = [ self.calculate(sample) for sample in getTestSet() ]
		return self.testPredictions

	def getTestClassPredictions(self):
		if self.testClassPredictions == None:
			self.testClassPredictions = [ 0 if v < 0.5 else 1 for v in self.getTestPredictions() ]
		return self.testClassPredictions



	def getTrainingRMSE(self):
		if self.trainingRMSE == None:
			pred = self.getTrainingPredictions()
			acc = 0
			ds = getTrainingSet()
			for i in range(len(ds)):
				dif = pred[i] - ds[i][-1]
				acc += dif**2
			acc /= len(ds)
			acc = acc**0.5
			self.trainingRMSE = acc 
		return self.trainingRMSE

	def getTestRMSE(self):
		if self.testRMSE == None:
			pred = self.getTestPredictions()
			acc = 0
			ds = getTestSet()
			for i in range(len(ds)):
				dif = pred[i] - ds[i][-1]
				acc += dif**2
			acc /= len(ds)
			acc = acc**0.5
			self.testRMSE = acc
		return self.testRMSE

	def getTrainingAccuracy(self):
		if self.trainingAccuracy == None:
			if OUTPUT != "Classification":
				self.trainingAccuracy = 0
			else:
				pred = self.getTrainingClassPredictions()
				hits = 0
				ds = getTrainingSet()
				for i in range(len(ds)):
					if pred[i] == ds[i][-1]:
						hits += 1
				acc = hits/len(ds)
				self.trainingAccuracy = acc
		return self.trainingAccuracy

	def getTestAccuracy(self):
		if self.testAccuracy == None:
			if OUTPUT != "Classification":
				self.testAccuracy = 0
			else:
				pred = self.getTestClassPredictions()
				hits = 0
				ds = getTestSet()
				for i in range(len(ds)):
					if pred[i] == ds[i][-1]:
						hits += 1
				acc = hits/len(ds)
				self.testAccuracy = acc
		return self.testAccuracy