from .Population import Population
from sklearn.tree import DecisionTreeClassifier
import random

# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#

class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """

    def __init__(self, expression, message = ""):
        self.expression = expression
        self.message = message


class MultiDT:
	population = None

	operators = None
	max_depth = None
	population_size = None
	max_generation = None
	tournament_size = None
	elitism_size = None
	limit_depth =None
	threads = None
	verbose = None


	def checkIfTrained(self):
		if self.best == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")


	def __init__(self, max_depth = 6, population_size = 100, limit_depth = 8, 
		threads=1, verbose = False):

		self.max_depth = max_depth
		self.population_size = population_size
		self.limit_depth = limit_depth
		self.threads = max(1, threads)
		self.verbose = verbose
		self.best = None
		pass

	def __str__(self):
		self.checkIfTrained()
		
		return str(self.getBestIndividual())
		

	def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
		num_features = Tr_X.shape[1]
		self.best = None
		self.best_fitness = -1
		self.best_tree_features = []
		for i in range(self.population_size):
			num_features_tree = random.randint(2, num_features)
			possible_features = [x for x in range(num_features)]
			features_tree = []
			for x in range(num_features_tree):
				next_feature = random.choice(possible_features)
				features_tree.append(next_feature)
				possible_features.remove(next_feature)
			tree_data = Tr_X.iloc[:, features_tree]
			tree = DecisionTreeClassifier(max_depth=6, criterion="entropy")
			tree.fit(tree_data, Tr_Y)
			if Te_X is not None:
				fitness = tree.score(Te_X, Te_Y)
			else:
				fitness = tree.score(tree_data, Tr_Y)
			if fitness > self.best_fitness or (fitness == self.best_fitness and len(features_tree) < len(self.best_tree_features)):
				self.best = tree
				self.best_fitness = fitness 
				self.best_tree_features = tree_data.columns
				#print(f"New best tree, fitness {self.best_fitness}, num_features {len(self.best_tree_features)}")


	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		self.checkIfTrained()

		return self.getBestIndividual().predict(dataset.loc[:, self.best_tree_features])

	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		self.checkIfTrained()

		return self.best

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		pass

	def getRMSEOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		pass

	def getF2OverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		pass 

	def getKappaOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		pass

	def getAUCOverTime(self):
		'''
		Returns the training and test AUC of the best model in each generation.
		'''
		pass

	def getSizeOverTime(self):
		'''
		Returns the size and number of dimensions of the best model in each generation.
		'''
		pass

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		pass

