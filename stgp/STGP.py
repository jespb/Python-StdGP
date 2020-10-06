from .Constants import *
from .Population import Population

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

class STGP:
	population = None

	def __init__(self, Tr, Te):
		setTerminals(Tr.columns[:-1])
		
		Tr = [ list(sample) for sample in Tr.iloc]
		Te = [ list(sample) for sample in Te.iloc]

		setTrainingSet(Tr)
		setTestSet(Te)

		self.population = Population()
		self.population.train()
		
	def getCurrentGeneration(self):
		'''
		Returns the number of the current generation.
		'''
		return self.population.getCurrentGeneration()

	def getTrainingAccuracy(self):
		'''
		Returns the training accuracy of the best individual
		'''
		return self.population.bestIndividual.getTrainingAccuracy()

	def getTestAccuracy(self):
		'''
		Returns the test accuracy of the best individual
		'''
		return self.population.bestIndividual.getTestAccuracy()
	
	def getTrainingRMSE(self):
		'''
		Returns the training rmse of the best individual
		'''
		return self.population.bestIndividual.getTrainingRMSE()
	
	def getTestRMSE(self):
		'''
		Returns the test rmse of the best individual
		'''
		return self.population.bestIndividual.getTestRMSE()
		
	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy over the generations
		'''
		return [self.population.trainingAccuracyOverTime, self.population.testAccuracyOverTime]

	def getRmseOverTime(self):
		'''
		Returns the training and test rmse over the generations
		'''
		return [self.population.trainingRmseOverTime, self.population.testRmseOverTime]

	def getSizeOverTime(self):
		'''
		Returns the size of the best individual over the generations
		'''
		return self.population.sizeOverTime 


	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		return self.population.getGenerationTimes()

	def getBestIndividual(self):
		'''
		Returns the best individual
		'''
		return self.population.bestIndividual