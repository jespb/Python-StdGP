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
		return self.getTrainingRMSE() < other.getTrainingRMSE()



	## FITNESS

	def getFitness(self):
		if self.fitness == None:
			self.fitness = self.getTrainingRMSE()
		return self.fitness

	def getTrainingRMSE(self):
		if self.trainingRMSE == None:
			acc = 0
			ds = getTrainingSet()
			for i in range(len(ds)):
				dif = self.predict(ds[i]) - ds[i][-1]
				acc += dif**2
			acc /= len(ds)
			acc = acc**0.5
			self.trainingRMSE = acc 
		return self.trainingRMSE

	def getTestRMSE(self):
		if self.testRMSE == None:
			acc = 0
			ds = getTestSet()
			for i in range(len(ds)):
				dif = self.predict(ds[i]) - ds[i][-1]
				acc += dif**2
			acc /= len(ds)
			acc = acc**0.5
			self.testRMSE = acc
		return self.testRMSE

	def getTrainingAccuracy(self):
		if self.trainingAccuracy == None:
			hits = 0
			ds = getTrainingSet()
			for i in range(len(ds)):
				if self.predict(ds[i]) == ds[i][-1]:
					hits += 1
			acc = hits/len(ds)
			self.trainingAccuracy = acc
		return self.trainingAccuracy

	def getTestAccuracy(self):
		if self.testAccuracy == None:
			hits = 0
			ds = getTestSet()
			for i in range(len(ds)):
				if self.predict(ds[i]) == ds[i][-1]:
					hits += 1
			acc = hits/len(ds)
			self.testAccuracy = acc
		return self.testAccuracy