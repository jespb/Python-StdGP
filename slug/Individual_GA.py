from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, cohen_kappa_score
from .STGP import STGP
from copy import deepcopy
import sys

class Individual_GA:
	training_X = None
	training_Y = None

	probabilities = None 
	GP_params = None

	fitness = None

	model_name = ["GP", "DT", "RF"][0]
	model = None

	fitnessType = ["Accuracy", "RMSE", "F2"][2]
	metrics = ["Acc", "Kappa", "F2", "AUC"]

	def __init__(self, probabilities, GP_params, metrics):
		self.probabilities = probabilities
		self.GP_params = GP_params
		self.metrics = metrics

	def create(self):
		if self.model_name == "GP":
			self.model = STGP(operators=self.GP_params["operators"],max_depth=self.GP_params["max_depth"], population_size=self.GP_params["population_size_GP"], max_generation=self.GP_params["max_generation_GP"],
			tournament_size=self.GP_params["tournament_size"], limit_depth=self.GP_params["limit_depth"], elitism_size=self.GP_params["elitism_size"],  threads=1, verbose=False)
		elif self.model_name == "DT":
			self.model = DecisionTreeClassifier()
		elif self.model_name == "RF":
			self.model = RandomForestClassifier()

	def copy(self, model, probabilities):
		self.model = model
		self.probabilities = probabilities


	def __gt__(self, other):
		sf = self.getFitness(self.fitnessType)

		of = other.getFitness(other.fitnessType)


		return (sf > of)

	def __ge__(self, other):
		return self.getFitness(self.fitnessType) >= other.getFitness(other.fitnessType)

	def __str__(self):
		return str(self.probabilities)


	def fit(self, Tr_x, Tr_y, Te_x, Te_y):

		tempTr = []
		tempTe = []
		for i in range(len(Tr_x.columns)):
			if self.probabilities[i] == 1:
				tempTr.append(Tr_x.iloc[:, i])
				tempTe.append(Te_x.iloc[:, i])
			else:
				pass
				
		Tr_x = pd.concat(tempTr, axis=1)
		Te_x = pd.concat(tempTe, axis=1)


		del tempTr
		del tempTe
		
		
		self.model.fit(Tr_x, Tr_y, Te_x, Te_y)


		# Obtain training results
		saved_metrics = []
		for metric in self.metrics:
			if metric == "Acc":
				saved_metrics.append(self.model.getAccuracyOverTime())
			elif metric == "F2":
				saved_metrics.append(self.model.getF2OverTime())
			elif metric == "Kappa":
				saved_metrics.append(self.model.getKappaOverTime())
			elif metric == "AUC":
				saved_metrics.append(self.model.getAUCOverTime())
		size      = self.model.getSizeOverTime()
		model_str = str(self.model.getBestIndividual())
		times     = self.model.getGenerationTimes()

		return (saved_metrics,
			size, times,
			model_str)





	def clone(self):
		
		return deepcopy(self)



	def getFitness(self, type):
		'''
		Returns the individual's fitness.
		'''
		self.fitness = self.model.getBestIndividual().fitness

		return self.fitness




