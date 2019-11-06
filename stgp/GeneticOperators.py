from .Constants import *
from .Individual import Individual
from .Node import Node
from random import random, randint

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

def getElite(population):
	return population[:ELITISM_SIZE]

def getOffspring(population):
	isCross = random()<0.9
	offspring = []
	if isCross:
		parents = [tournament(population),tournament(population)]

		osxo = crossover(parents)
		
		isMutation = random() < 0.1
		if isMutation:
			for i in range(len(osxo)):
				osxom = mutation(osxo[i])
				offspring.extend(osxom)
		else:
			offspring.extend( osxo )
	
	else:
		parent = tournament(population)
		isMutation = random() < 0.1
		if isMutation:
			osm = mutation(parent)
			offspring.extend(osm)
		else:
			offspring.append(parent)
	
	return offspring

def tournament(population):
	candidates = [randint(0,len(population)-1) for i in range(TOURNAMENT_SIZE)]
	return population[min(candidates)]

def crossover(parents):
	ind1 = parents[0]
	ind2 = parents[1]
	n1 = ind1.getHead()
	n2 = ind2.getHead()
	n11 = n1.getRandomNode()
	n21 = n2.getRandomNode()
	n11.swap(n21)

	ret = [Individual(n1), Individual(n2)]

	# Rejects indivials over a certain depth
	i = 0
	while i < len(ret):
		if ret[i].getDepth() > LIMIT_DEPTH:
			ret.pop(i)
			i-=1
		i+=1

	return ret

def mutation(parent):
	ind1 = parent
	n1 = ind1.getHead()
	n11 = n1.getRandomNode()
	n11.swap(Node())
	
	ret = [Individual(n1)]
	
	# Rejects indivials over a certain depth
	i = 0
	while i < len(ret):
		if ret[i].getDepth() > LIMIT_DEPTH:
			ret.pop(i)
			i-=1
		i+=1

	return ret

