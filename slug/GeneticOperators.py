from .Individual import Individual
from .Node import Node
from random import random, randint
import numpy as np

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#

def roulette(population):
  
    # Computes the totallity of the features fitness
    max = sum([indiv.getFitness('Accuracy') for indiv in population])
    
    # Computes for each feature the probability 
    indiv_probabilities = [indiv.getFitness('Accuracy')/max for indiv in population]
	
    return population[np.random.choice(len(population), p=indiv_probabilities)]

def GAoffspring(population):
	crossprob = random()
	mutprob = random()

	p1, p2 = roulette(population), roulette(population)

	if crossprob <= 0.7:
		p1, p2 = GAcrossover(p1,p2)

	if mutprob <= 1/len(population):
		p1 = GAmutation(p1)
	if mutprob <= 1/len(population):
		p2 = GAmutation(p2)

	return p1, p2


def GAcrossover(p1, p2):
	cut = randint(1,len(p1.probabilities))

	temp = p1.probabilities[:-cut]
	temp.extend(p2.probabilities[-cut:])

	new_p1 = p1.clone()
	new_p1.probabilities = fixAllZeros(temp, len(temp))

	temp = p2.probabilities[:-cut]
	temp.extend(p1.probabilities[-cut:])

	new_p2 = p2.clone()
	new_p2.probabilities = fixAllZeros(temp, len(temp))

	#print('p1 crossover: ', new_p1.probabilities)
	#print('p2 crossover: ', new_p2.probabilities)

	return new_p1, new_p2


def GAmutation(indiv):

	probs_flipped = [1-value if random() <= 1/len(indiv.probabilities) else value for value in indiv.probabilities]
	indiv.probabilities = fixAllZeros(probs_flipped, len(probs_flipped))


	#print('mutation: ', indiv.probabilities)

	return indiv

def fixAllZeros(l, n):
	while np.all((np.array(l) == 0)):
		l = [randint(0,1) for col in range(n)]

	return l


def tournament(population,n):
	'''
	Selects "n" Individuals from the population and return a 
	single Individual.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	candidates = [randint(0,len(population)-1) for i in range(n)]
	return population[min(candidates)]


def getElite(population,n):
	'''
	Returns the "n" best Individuals in the population.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	return population[:n]


def getOffspring(population, tournament_size):
	'''
	Genetic Operator: Selects a genetic operator and returns a list with the 
	offspring Individuals. The crossover GOs return two Individuals and the
	mutation GO returns one individual.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	isCross = random()<0.5
	desc = None

	if isCross:
		desc = STXO(population, tournament_size)
	else:
		desc = STMUT(population, tournament_size)

	return desc


def discardDeep(population, limit):
	ret = []
	for ind in population:
		if ind.getDepth() <= limit:
			ret.append(ind)
	return ret


def STXO(population, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = tournament(population, tournament_size)
	ind2 = tournament(population, tournament_size)

	d1 = ind1.head.clone()
	d2 = ind2.head.clone()

	n1 = d1.getRandomNode()
	n2 = d2.getRandomNode()

	n1.swap(n2)

	ret = []
	for d in [d1,d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth)
		i.copy(d)
		ret.append(i)
	return ret



def STMUT(population, tournament_size):
	'''
	Randomly selects one node from a single individual; swaps the node with a 
	new, node generated using Grow; and returns the new Individual as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1 = tournament(population, tournament_size)
	d1 = ind1.head.clone()
	n1 = d1.getRandomNode()
	n = Node()
	n.create(ind1.operators, ind1.terminals, ind1.max_depth)
	n1.swap(n)


	ret = []
	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth)
	i.copy(d1)
	ret.append(i)
	return ret

