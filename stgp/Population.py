from .Individual import Individual
from .Node import Node
from .Constants import *
from .GeneticOperators import getElite, getOffspring
from random import random, randint

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

class Population:
	population = None
	bestIndividual = None
	output = None

	trainingOverTime = None
	testOverTime = None
	sizeOverTime = None
	currentGeneration = None


	def __init__(self, output="Classification"):
		self.output = output
		self.currentGeneration = 0
		self.population = []
		self.trainingOverTime = []
		self.testOverTime = []
		self.sizeOverTime = []
		for i in range(POPULATION_SIZE):
			self.population.append(Individual())


	def stoppingCriteria(self):
		genLimit = self.currentGeneration >= MAX_GENERATION
		perfectTraining = self.bestIndividual != None
		perfectTraining = perfectTraining and self.bestIndividual.getTrainingRMSE() == 0
		
		return genLimit or perfectTraining


	def train(self):
		while not self.stoppingCriteria():
			self.nextGeneration()
			self.currentGeneration += 1
			#Mudar isto para tambem fazer o RMSE
			self.trainingOverTime.append(self.bestIndividual.getTrainingAccuracy())
			self.testOverTime.append(self.bestIndividual.getTestAccuracy())
			self.sizeOverTime.append(self.bestIndividual.getSize())
		while self.currentGeneration < MAX_GENERATION:
			self.currentGeneration += 1
			self.trainingOverTime.append(self.bestIndividual.getTrainingAccuracy())
			self.testOverTime.append(self.bestIndividual.getTestAccuracy())
			self.sizeOverTime.append(self.bestIndividual.getSize())		


	def nextGeneration(self):
		# Calculates Fitness (RMSE)
		fitness = []
		for individual in self.population:
			fitness.append(individual.getFitness())
			#print(individual.getFitness())
			
		# Sort the population from best to worse
		self.population.sort(reverse=True) 

		# Update Best Individual
		if(self.bestIndividual == None or self.population[0]>self.bestIndividual):
			self.bestIndividual = self.population[0]

		if self.output == "Classification":
			print("Gen#",self.currentGeneration, "- (TrA,TeA,TrRMSE):", self.bestIndividual.getTrainingAccuracy(),self.bestIndividual.getTestAccuracy(),self.bestIndividual.getTrainingRMSE())
		if self.output == "Regression":
			print("Gen#",self.currentGeneration, "- (TrRMSE,TeRMSE):", self.bestIndividual.getTrainingRMSE(), self.bestIndividual.getTestRMSE())
		
		# Generating Next Generation
		newPopulation = []
		newPopulation.extend( getElite(self.population) )
		while len(newPopulation) < POPULATION_SIZE:
			newPopulation.extend( getOffspring(self.population) )
		self.population = newPopulation[:POPULATION_SIZE]

	def predict(self, sample):
		return "Population Not Trained" if self.bestIndividual == None else self.bestIndividual.predict(sample)

	def getBestIndividual(self):
		return self.bestIndividual

	def getCurrentGeneration(self):
		return self.currentGeneration

