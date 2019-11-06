# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

OPERATORS = ["+","-","*","/"]
MAX_DEPTH = 6 # max depth of the initial trees and the trees used for mutation
POPULATION_SIZE = 200
MAX_GENERATION = 10
TRAIN_FRACTION = 0.70
TOURNAMENT_SIZE = 10
ELITISM_SIZE = 1
SHUFFLE = True
LIMIT_DEPTH=15
RUNS = 10

out = None

def openFile(name):
	global out
	out = open(name,"w")

def writeToFile(msg):
	global out
	out.write(msg)

def closeFile():
	global out
	out.close()

terminals = None
def setTerminals(l):
	global terminals 
	terminals = l
def getTerminals():
	return terminals

trainingSet = None
def setTrainingSet(ds):
	global trainingSet
	trainingSet = ds
def getTrainingSet():
	return trainingSet

testSet = None
def setTestSet(ds):
	global testSet
	testSet = ds
def getTestSet():
	return testSet