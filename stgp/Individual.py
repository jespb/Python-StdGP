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

	fitness = None

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



	def setFitness(self, fitness):
		self.fitness = fitness

	def setTrainingRMSE(self, rmse):
		self.trainingRMSE = rmse

	def setTestRMSE(self, rmse):
		self.testRMSE = rmse

	def setTrainingAccuracy(self, accuracy):
		self.trainingAccuracy = accuracy

	def setTestAccuracy(self, accuracy):
		self.testAccuracy = accuracy



	def __str__(self):
		return str(self.head)

	def __lt__(self, other):
		# Using RMSE as fitness
		return self.fitness > other.fitness



	## FITNESS

	def getFitness(self):
		if self.fitness == None:
			self.fitness = self.getTrainingRMSE()
		return self.fitness

	def getTrainingRMSE(self):
		if self.trainingRMSE != None:
			return self.trainingRMSE
		acc = 0
		ds = getTrainingSet()
		for i in range(len(ds)):
			dif = self.predict(ds[i]) - ds[i][-1]
			acc += dif**2
		acc /= len(ds)
		acc = acc**0.5
		self.setTrainingRMSE(acc)
		return acc

	def getTestRMSE(self):
		if self.testRMSE != None:
			return self.testRMSE
		acc = 0
		ds = getTestSet()
		for i in range(len(ds)):
			dif = self.predict(ds[i]) - ds[i][-1]
			acc += dif**2
		acc /= len(ds)
		acc = acc**0.5
		self.setTestRMSE(acc)
		return acc

	def getTrainingAccuracy(self):
		if self.trainingAccuracy != None:
			return self.trainingAccuracy
		hits = 0
		ds = getTrainingSet()
		for i in range(len(ds)):
			if self.predict(ds[i]) == ds[i][-1]:
				hits += 1
		acc = hits/len(ds)
		self.setTrainingAccuracy(acc)
		return acc

	def getTestAccuracy(self):
		if self.testAccuracy != None:
			return self.testAccuracy
		hits = 0
		ds = getTestSet()
		for i in range(len(ds)):
			if self.predict(ds[i]) == ds[i][-1]:
				hits += 1
		acc = hits/len(ds)
		self.setTestAccuracy(acc)
		return acc