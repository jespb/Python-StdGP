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

	def __init__(self, panda_ds):
		terminals = list(panda_ds.columns[:-1])
		setTerminals(terminals)

		if SHUFFLE:
			panda_ds = panda_ds.sample(frac=1)
		train_ds_size = int(panda_ds.shape[0]*TRAIN_FRACTION)
		train_ds = []
		for i in range(train_ds_size):
			train_ds.append(list(panda_ds.iloc[i]))
		test_ds = []
		for i in range(train_ds_size, panda_ds.shape[0]):
			test_ds.append(list(panda_ds.iloc[i]))
		setTrainingSet(train_ds)
		setTestSet(test_ds)

		self.population = Population()
		self.population.train()
		
	def getCurrentGeneration(self):
		return self.population.getCurrentGeneration()

	def getTrainingAccuracy(self):
		return self.population.bestIndividual.getTrainingAccuracy() if self.output == "Classification" else 0

	def getTestAccuracy(self):
		return self.population.bestIndividual.getTestAccuracy() if self.output == "Classification" else 0
	
	def getTrainingRMSE(self):
		return self.population.bestIndividual.getTrainingRMSE()
	
	def getTestRMSE(self):
		return self.population.bestIndividual.getTestRMSE()
		
	def getAccuracyOverTime(self):
		return [self.population.trainingAccuracyOverTime, self.population.testAccuracyOverTime]

	def getRmseOverTime(self):
		return [self.population.trainingRmseOverTime, self.population.testRmseOverTime]

	def getSizeOverTime(self):
		return self.population.sizeOverTime 

	def getBestIndividual(self):
		return self.population.bestIndividual