from .Individual import Individual
from .Node import Node
from random import random, randint

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#


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

