from .Population_GA import Population_GA


class GA:
	population = None
	population_size = None
	max_generation = None
	elitism_size = None
	verbose = None



	def __init__(self, population_size = 100, max_generation = 50, elitism_size = 1, metrics = ["Acc", "Kappa", "F2", "AUC"],
				GP_params={"population_size_gp":100, "max_generation_GP":30, "operators":[("+",2),("-",2),("*",2),("/",2)], "max_depth":6, "limit_depth":17, "elitism_size":1},
				threads=1, verbose = False):
 
		self.population_size = population_size
		self.max_generation = max_generation
		self.elitism_size = elitism_size
		self.metrics = metrics
		self.GP_params = GP_params
		self.verbose = verbose
		self.threads = threads

	def __str__(self):
		
		return str(self.getBestIndividual())
		

	def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
		if self.verbose:
			print("Training a model with the following parameters: ", end="")
			print("{Population Size : "+str(self.population_size)+"}, ", end="")
			print("{Max Generation : "+str(self.max_generation)+"}, ", end="")
			#print("{Number of Blocks : "+str(self.n_blocks)+"}, ", end="")
			print("{Elitism Size : "+str(self.elitism_size)+"}, ", end="")
	

		self.population = Population_GA(Tr_X, Tr_Y, Te_X, Te_Y, self.population_size, self.max_generation, self.elitism_size, self.metrics, self.GP_params, self.threads)
		self.population.train()

		self.getBestIndividual()

	def predict(self, dataset):
		'''
		Returns the predictions for the samples in a dataset.
		'''
		#self.checkIfTrained()

		return self.population.getBestIndividual().predict(dataset)

	def getBestIndividual(self):
		'''
		Returns the final M3GP model.
		'''
		#self.checkIfTrained()

		return self.population.getBestIndividual()

	def getAccuracyOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		#self.checkIfTrained()

		return [self.population.getTrainingAccuracyOverTime(), self.population.getTestAccuracyOverTime()]

	def getF2OverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		#self.checkIfTrained()

		return [self.population.getTrainingF2OverTime(), self.population.getTestF2OverTime()]

	def getKappaOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		#self.checkIfTrained()

		return [self.population.getTrainingKappaOverTime(), self.population.getTestKappaOverTime()]

	def getAUCOverTime(self):
		'''
		Returns the training and test AUC of the best model in each generation.
		'''
		#self.checkIfTrained()

		return [self.population.getTrainingAUCOverTime(), self.population.getTestAUCOverTime()]

	def getRMSEOverTime(self):
		'''
		Returns the training and test accuracy of the best model in each generation.
		'''
		#self.checkIfTrained()

		return [self.population.getTrainingRMSEOverTime(), self.population.getTestRMSEOverTime()]


		return self.population.getSizeOverTime()

	def getGenerationTimes(self):
		'''
		Returns the time spent in each generation.
		'''
		#self.checkIfTrained()

		return self.population.getGenerationTimes()


	def getFeaturesOverTime(self):
	

		return self.population.getFeaturesOverTime()
