from .Individual_GA import Individual_GA
from .GeneticOperators import GAoffspring, getElite
import multiprocessing as mp
import time
import random
import numpy as np

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/NMVRodrigues/SLUG
#
# Copyright ©2021-2022 N. M. Rodrigues
#

class Population_GA:
	population_size = None
	max_generation = None
	elitism_size = None


	population = None
	bestIndividual = None
	currentGeneration = 0

	trainingF2OverTime = None
	testF2OverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	sizeOverTime = None
	featuresOverTime = None

	generationTimes = None


	def __init__(self, Tr_x, Tr_y, Te_x, Te_y, population_size, max_generation, elitism_size, metrics, GP_params, classifier, threads):

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y

		self.terminals = list(Tr_x.columns)
		self.population_size = population_size
		self.max_generation = max_generation
		self.elitism_size = elitism_size
		self.GP_params = GP_params
		self.classifier = classifier

		self.population = []
		self.threads = threads
		self.metrics = metrics
		self.verbose = True

		while len(self.population) < self.population_size:
			probs = [random.randint(0,1) for col in self.terminals]
			probs = fixAllZeros(probs, len(self.terminals))
			ind = Individual_GA(probs, self.GP_params, self.metrics, self.classifier)
			ind.create()
			self.population.append(ind)

		self.bestIndividual = self.population[0]
		self.bestIndividual.fit(self.Tr_x, self.Tr_y, self.Te_x, self.Te_y)


		if not self.Te_x is None:
			if "Acc" in self.metrics:
				self.trainingAccuracyOverTime = []
				self.testAccuracyOverTime = []
			if "F2" in self.metrics:
				self.trainingF2OverTime = []
				self.testF2OverTime = []
			if "Kappa" in self.metrics:
				self.trainingKappaOverTime = []
				self.testKappaOverTime = []
			if "AUC" in self.metrics:
				self.trainingAUCOverTime = []
				self.testAUCOverTime = []
			self.sizeOverTime = []
			self.generationTimes = []
			self.featuresOverTime = []


	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation
		
		return genLimit


	def train(self):
		'''
		Training loop for the algorithm.
		'''
		if self.verbose:
			print("> Running log:")

		while self.currentGeneration < self.max_generation:
			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			else:
				duration = 0
			self.currentGeneration += 1
			
			if not self.Te_x is None:
				if self.classifier == 'GP':
					if "Acc" in self.metrics:
						self.trainingAccuracyOverTime.append(self.bestIndividual.model.getBestIndividual().getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testAccuracyOverTime.append(self.bestIndividual.model.getBestIndividual().getAccuracy(self.Te_x, self.Te_y, pred="Te"))
					if "F2" in self.metrics:
						self.trainingF2OverTime.append(self.bestIndividual.model.getBestIndividual().getF2(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testF2OverTime.append(self.bestIndividual.model.getBestIndividual().getF2(self.Te_x, self.Te_y, pred="Te"))
					if "Kappa" in self.metrics:
						self.trainingKappaOverTime.append(self.bestIndividual.model.getBestIndividual().getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testKappaOverTime.append(self.bestIndividual.model.getBestIndividual().getKappa(self.Te_x, self.Te_y, pred="Te"))
					if "AUC" in self.metrics:
						self.trainingAUCOverTime.append(self.bestIndividual.model.getBestIndividual().getAUC(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testAUCOverTime.append(self.bestIndividual.model.getBestIndividual().getAUC(self.Te_x, self.Te_y, pred="Te"))
					self.featuresOverTime.append(self.bestIndividual.probabilities)
					self.sizeOverTime.append(self.bestIndividual.model.getBestIndividual().getSize())
					self.generationTimes.append(duration)
				else:
					if "Acc" in self.metrics:
						self.trainingAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testAccuracyOverTime.append(self.bestIndividual.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
					if "F2" in self.metrics:
						self.trainingF2OverTime.append(self.bestIndividual.getF2(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testF2OverTime.append(self.bestIndividual.getF2(self.Te_x, self.Te_y, pred="Te"))
					if "Kappa" in self.metrics:
						self.trainingKappaOverTime.append(self.bestIndividual.getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testKappaOverTime.append(self.bestIndividual.getKappa(self.Te_x, self.Te_y, pred="Te"))
					if "AUC" in self.metrics:
						self.trainingAUCOverTime.append(self.bestIndividual.getAUC(self.Tr_x, self.Tr_y, pred="Tr"))
						self.testAUCOverTime.append(self.bestIndividual.getAUC(self.Te_x, self.Te_y, pred="Te"))
					self.featuresOverTime.append(self.bestIndividual.probabilities)
					self.sizeOverTime.append(None)
					self.generationTimes.append(duration)

		if self.verbose:
			print()



	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()
		
		if self.threads > 1:
			with mp.Pool(processes= self.threads) as pool:
				model = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y, self.Te_x, self.Te_y) for ind in self.population] )
				for i in range(len(self.population)):
					#print('model: ', model)
					self.population[i].model = model[i][0]
					self.population[i].fitness = model[i][1]
					#self.population[i].labelToInt = model[i].labelToInt
					#self.population[i].intToLabel = model[i].intToLabel
					#self.population[i].trainingPredictions = model[i][1]
					#self.population[i].probabilities = model[i].probabilities
					#self.population[i].training_X = self.Tr_x
					#self.population[i].training_Y = self.Tr_y
		else:
			[ ind.fit(self.Tr_x, self.Tr_y, self.Te_x, self.Te_y) for ind in self.population]
			[ ind.getFitness() for ind in self.population ]

		fit = []
		for ind in self.population:
			fit.append(ind.fitness)
		#print(min(fit), max(fit)) # *** J: as DT parece estagnar bem depressa na accuracy maxima da geracao mas a minima vai subido hmmm


		# Sort the population from best to worse
		self.population.sort(reverse=True)


		# Update best individual
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]

		# Generating Next Generation
		newPopulation = []
		#newPopulation.extend(getElite(self.population, self.elitism_size))
		while len(newPopulation) < self.population_size:
			offspring = GAoffspring(self.population)
			newPopulation.extend(offspring)
		self.population = newPopulation[:self.population_size]


		end = time.time()

		# Debug
		if self.verbose and self.currentGeneration%5==0:
			if self.classifier == 'GP':
				if not self.Te_x is None:
					print("> Gen #"+str(self.currentGeneration)+":   Tr-Acc: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getAccuracy(self.Tr_x, self.Tr_y)+" // Te-Acc: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getAccuracy(self.Te_x, self.Te_y) +
					":  Tr-F2: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getF2(self.Tr_x, self.Tr_y)+" // Te-F2: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getF2(self.Te_x, self.Te_y) +
					":  Tr-Kappa: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getKappa(self.Tr_x, self.Tr_y)+" // Te-Kappa: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getKappa(self.Te_x, self.Te_y) +
					":  Tr-ROC_AUC: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getAUC(self.Tr_x, self.Tr_y)+" // Te-ROC_AUC: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getAUC(self.Te_x, self.Te_y) + " // Time: " + str(end- begin) , '\n')
				else:
					print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.bestIndividual.model.getBestIndividual().getAccuracy(self.Tr_x, self.Tr_y))
			else:
				if not self.Te_x is None:
					print("> Gen #"+str(self.currentGeneration)+":   Tr-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y)+" // Te-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Te_x, self.Te_y) +
					":  Tr-F2: "+ "%.6f" %self.bestIndividual.getF2(self.Tr_x, self.Tr_y)+" // Te-F2: "+ "%.6f" %self.bestIndividual.getF2(self.Te_x, self.Te_y) +
					":  Tr-Kappa: "+ "%.6f" %self.bestIndividual.getKappa(self.Tr_x, self.Tr_y)+" // Te-Kappa: "+ "%.6f" %self.bestIndividual.getKappa(self.Te_x, self.Te_y) +
					":  Tr-ROC_AUC: "+ "%.6f" %self.bestIndividual.getAUC(self.Tr_x, self.Tr_y)+" // Te-ROC_AUC: "+ "%.6f" %self.bestIndividual.getAUC(self.Te_x, self.Te_y) + " // Time: " + str(end- begin) , '\n')
				else:
					print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y))


	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)

	def getBestIndividual(self):
		return self.bestIndividual

	def getCurrentGeneration(self):
		return self.currentGeneration

	def getTrainingAccuracyOverTime(self):
		return self.trainingAccuracyOverTime

	def getTestAccuracyOverTime(self):
		return self.testAccuracyOverTime

	def getTrainingRMSEOverTime(self):
		return self.trainingRMSEOverTime

	def getTestRMSEOverTime(self):
		return self.testRMSEOverTime

	def getTrainingF2OverTime(self):
		return self.trainingF2OverTime

	def getTestF2OverTime(self):
		return self.testF2OverTime

	def getTrainingKappaOverTime(self):
		return self.trainingKappaOverTime

	def getTestKappaOverTime(self):
		return self.testKappaOverTime

	def getTrainingAUCOverTime(self):
		return self.trainingAUCOverTime

	def getTestAUCOverTime(self):
		return self.testAUCOverTime

	def getGenerationTimes(self):
		return self.generationTimes

	def getFeaturesOverTime(self):
		return self.featuresOverTime


def fitIndividuals(a):
	ind,x,y,tx, ty = a
	ind.fit(x,y, tx, ty)
	ind.getFitness(x,y)
	
	return (ind.model, ind.fitness)

def fixAllZeros(l, n):
	while np.all((np.array(l) == 0)):
		l = [random.randint(0,1) for col in range(n)]

	return l