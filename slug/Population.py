from .Individual import Individual
from .GeneticOperators import getElite, getOffspring, discardDeep
import multiprocessing as mp
import time

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#

class Population:
	operators = None
	max_initial_depth = None
	population_size = None
	max_generation = None
	tournament_size = None
	elitism_size = None
	limit_depth = None
	verbose = None
	threads = None
	terminals = None
	features = None


	population = None
	bestIndividual = None
	currentGeneration = 0

	trainingF2OverTime = None
	testF2OverTime = None
	trainingKappaOverTime = None
	testKappaOverTime = None
	sizeOverTime = None

	generationTimes = None


	def __init__(self, Tr_x, Tr_y, Te_x, Te_y, operators, max_initial_depth, population_size,
		max_generation, tournament_size, elitism_size, limit_depth, threads, verbose, metrics = ["Acc", "Kappa", "F2", "AUC"]):

		self.Tr_x = Tr_x
		self.Tr_y = Tr_y
		self.Te_x = Te_x
		self.Te_y = Te_y

		self.terminals = list(Tr_x.columns)
		self.operators = operators
		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.max_generation = max_generation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size
		self.limit_depth = limit_depth
		self.threads = threads
		self.verbose = verbose

		self.metrics = metrics

		self.population = []

		while len(self.population) < self.population_size:
			ind = Individual(self.operators, self.terminals, self.max_initial_depth)
			ind.create()
			self.population.append(ind)

		self.bestIndividual = self.population[0]
		self.bestIndividual.fit(self.Tr_x, self.Tr_y)


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


	def stoppingCriteria(self):
		'''
		Returns True if the stopping criteria was reached.
		'''
		genLimit = self.currentGeneration >= self.max_generation
		perfectTraining = self.bestIndividual.getFitness() == 1
		
		return genLimit  or perfectTraining


	def train(self):
		'''
		Training loop for the algorithm.
		'''
		#if self.verbose:
			#print("> Running log:")

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
				self.sizeOverTime.append(self.bestIndividual.getSize())
				self.generationTimes.append(duration)

		if self.verbose:
			#print()
			pass



	def nextGeneration(self):
		'''
		Generation algorithm: the population is sorted; the best individual is pruned;
		the elite is selected; and the offspring are created.
		'''
		begin = time.time()
		
		# Calculates the accuracy of the population using multiprocessing
		if self.threads > 1:
			with mp.Pool(processes= self.threads) as pool:
				model = pool.map(fitIndividuals, [(ind, self.Tr_x, self.Tr_y) for ind in self.population] )
				for i in range(len(self.population)):
					print('1: ', self.population[i])
					print('2: ', self.population[i].model)
					print('3: ', model[i][0])
					print('4: ', model)
					
					self.population[i].model = model[i][0].model
					self.population[i].labelToInt = model[i][0].labelToInt
					self.population[i].intToLabel = model[i][0].intToLabel
					self.population[i].trainingPredictions = model[i][1]
					self.population[i].training_X = self.Tr_x
					self.population[i].training_Y = self.Tr_y
		else:
			[ ind.fit(self.Tr_x, self.Tr_y) for ind in self.population]
			[ ind.getFitness() for ind in self.population ]

		# Sort the population from best to worse
		self.population.sort(reverse=True)


		# Update best individual
		if self.population[0] > self.bestIndividual:
			self.bestIndividual = self.population[0]

		# Generating Next Generation
		newPopulation = []
		newPopulation.extend(getElite(self.population, self.elitism_size))
		while len(newPopulation) < self.population_size:
			offspring = getOffspring(self.population, self.tournament_size)
			offspring = discardDeep(offspring, self.limit_depth)
			newPopulation.extend(offspring)
		self.population = newPopulation[:self.population_size]


		end = time.time()

		# Debug
		#if self.verbose and self.currentGeneration%5==0:
		#if self.verbose and self.currentGeneration+1==self.max_generation:
	#		if not self.Te_x is None:
	#			print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y)+" // Te-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Te_x, self.Te_y) + " // Time: " + str(end- begin) )
#			else:
				#print("   > Gen #"+str(self.currentGeneration)+":  Tr-Acc: "+ "%.6f" %self.bestIndividual.getAccuracy(self.Tr_x, self.Tr_y))


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

	def getSizeOverTime(self):
		return self.sizeOverTime

	def getGenerationTimes(self):
		return self.generationTimes


def calculateIndividual_MultiProcessing(ind, fitArray, indIndex):
	fitArray[indIndex] = ind.getTrainingF2()

def fitIndividuals(a):
	ind,x,y = a
	ind.fit(x,y)
	
	return ( ind, ind.predict(x) )

def getTrainingPredictions(ind):
	return ind.getTrainingPredictions()
