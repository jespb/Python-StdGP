from .Individual import Individual
from .Node import Node
from .Constants import *
from .GeneticOperators import getElite, getOffspring
from random import random, randint
import time

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

	trainingAccuracyOverTime = None
	testAccuracyOverTime = None
	trainingRmseOverTime = None
	testRmseOverTime = None
	sizeOverTime = None
	currentGeneration = None
	generationTime = None


	def __init__(self):
		self.currentGeneration = 0
		self.population = []
		self.trainingAccuracyOverTime = []
		self.testAccuracyOverTime = []
		self.trainingRmseOverTime = []
		self.testRmseOverTime = []
		self.sizeOverTime = []
		self.generationTimes = []
		for i in range(POPULATION_SIZE):
			self.population.append(Individual())


	def stoppingCriteria(self):
		genLimit = self.currentGeneration >= MAX_GENERATION
		perfectTraining = self.bestIndividual != None
		perfectTraining = perfectTraining and self.bestIndividual.getTrainingRMSE() == 0
		
		return genLimit or perfectTraining


	def train(self):
		while self.currentGeneration < MAX_GENERATION:
			duration = 0

			if not self.stoppingCriteria():
				t1 = time.time()
				self.nextGeneration()
				t2 = time.time()
				duration = t2-t1
			
			self.currentGeneration += 1
			self.trainingAccuracyOverTime.append(self.bestIndividual.getTrainingAccuracy())
			self.testAccuracyOverTime.append(self.bestIndividual.getTestAccuracy())
			self.trainingRmseOverTime.append(self.bestIndividual.getTrainingRMSE())
			self.testRmseOverTime.append(self.bestIndividual.getTestRMSE())
			self.sizeOverTime.append(self.bestIndividual.getSize())		
			self.generationTimes.append(duration)	



	def nextGeneration(self):
		# Sort the population from best to worse
		self.population.sort(reverse=True) 

		# Update Best Individual
		if(self.bestIndividual == None or self.population[0]>self.bestIndividual):
			self.bestIndividual = self.population[0]

		if self.currentGeneration%10 == 0:
			if OUTPUT == "Classification":
				print("Gen#",self.currentGeneration, "- (TrA,TeA,TrRMSE):", self.bestIndividual.getTrainingAccuracy(),self.bestIndividual.getTestAccuracy(),self.bestIndividual.getTrainingRMSE())
			if OUTPUT == "Regression":
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

	def getGenerationTimes(self):
		return self.generationTimes

