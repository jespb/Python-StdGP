from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, cohen_kappa_score, roc_auc_score
from .STGP import STGP
from copy import deepcopy
import sys

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/NMVRodrigues/SLUG
#
# Copyright ©2021-2022 N. M. Rodrigues
#

class Individual_GA:
	training_X = None
	training_Y = None

	probabilities = None 
	GP_params = None

	fitness = None

	#model_name = ["GP", "DT", "RF"][1]
	model = None

	fitnessType = ["Accuracy", "RMSE", "F2", "2FOLD-ACC"][0]
	metrics = ["Acc", "Kappa", "F2", "AUC"]

	def __init__(self, probabilities, GP_params, metrics, classifier):
		self.probabilities = probabilities
		self.GP_params = GP_params
		self.metrics = metrics
		self.classifier = classifier

	def create(self):
		if self.classifier == "GP":
			self.model = STGP(operators=self.GP_params["operators"],max_depth=self.GP_params["max_depth"], population_size=self.GP_params["population_size_GP"], max_generation=self.GP_params["max_generation_GP"],
			tournament_size=self.GP_params["tournament_size"], limit_depth=self.GP_params["limit_depth"], elitism_size=self.GP_params["elitism_size"],  threads=1, verbose=False)
		elif self.classifier == "DT":
			self.model = DecisionTreeClassifier(random_state=42, max_depth=2) 
		elif self.classifier == "RF":
			self.model = RandomForestClassifier(random_state=42) 
		elif self.classifier == "SVM":
			self.model =  SVC(random_state=42) 

	# for 2fold-acc
	def create_tmp_model(self):
		if self.classifier == "GP":
			return STGP(operators=self.GP_params["operators"],max_depth=self.GP_params["max_depth"], population_size=self.GP_params["population_size_GP"], max_generation=self.GP_params["max_generation_GP"],
			tournament_size=self.GP_params["tournament_size"], limit_depth=self.GP_params["limit_depth"], elitism_size=self.GP_params["elitism_size"],  threads=1, verbose=False)
		elif self.classifier == "DT":
			return DecisionTreeClassifier(random_state=42) 
		elif self.classifier == "RF":
			return RandomForestClassifier(random_state=42) 
		elif self.classifier == "SVM":
			self.model =  SVC(random_state=42)

	def copy(self, model, probabilities):
		self.model = model
		self.probabilities = probabilities


	def __gt__(self, other):
		sf = self.getFitness(self.training_X, self.training_Y)
		of = other.getFitness(other.training_X, other.training_Y)

		# NEW BLOAT CONTROL
		ss = sum(self.probabilities)
		os = sum(other.probabilities)


		return (sf > of) or (sf==of and ss<os)

	def __ge__(self, other):
		return self.getFitness() >= other.getFitness()

	def __str__(self):
		return str(self.probabilities)


	def fit(self, Tr_x, Tr_y, Te_x, Te_y):
		self.training_X = Tr_x 
		self.training_Y = Tr_y

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
		
		if self.classifier == 'GP':
			self.model.fit(Tr_x, Tr_y.values, Te_x, Te_y.values)

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

		else:
			self.model.fit(Tr_x, Tr_y)
			y_pred = self.model.predict(Tr_x)
			y_pred_test = self.model.predict(Te_x)

			# Obtain training results
			saved_metrics = []
			for metric in self.metrics:
				if metric == "Acc":
					saved_metrics.append([accuracy_score(Tr_y, y_pred), accuracy_score(Te_y, y_pred_test)])
				elif metric == "F2":
					saved_metrics.append([fbeta_score(Tr_y, y_pred, beta=2), fbeta_score(Te_y, y_pred_test, beta=2)])
				elif metric == "Kappa":
					saved_metrics.append([cohen_kappa_score(Tr_y, y_pred), cohen_kappa_score(Te_y, y_pred_test)])
				elif metric == "AUC":
					saved_metrics.append([roc_auc_score(Tr_y, y_pred), roc_auc_score(Te_y, y_pred_test)])
			size      = [None]
			model_str = [None]
			times     = [None]


		return (saved_metrics,
			size, times,
			model_str)



	def clone(self):
		
		return deepcopy(self)



	def getFitness(self, Tr_x = None, Tr_y=None):
		'''
		Returns the individual's fitness.
		'''
		if self.fitness != None:
			return self.fitness

		if self.classifier == 'GP':
			self.fitness = self.model.getBestIndividual().fitness
		else:
			if self.fitnessType == "Accuracy":
				Tr_x = self.training_X
				Tr_y = self.training_Y

				tempTr = []
				for i in range(len(Tr_x.columns)):
					if self.probabilities[i] == 1:
						tempTr.append(Tr_x.iloc[:, i])
				
				Tr_x = pd.concat(tempTr, axis=1)
				del tempTr

				pred = self.model.predict(Tr_x)
				self.fitness = accuracy_score(Tr_y, pred)

			if self.fitnessType == "F2":
				Tr_x = self.training_X
				Tr_y = self.training_Y

				tempTr = []
				for i in range(len(Tr_x.columns)):
					if self.probabilities[i] == 1:
						tempTr.append(Tr_x.iloc[:, i])
				
				Tr_x = pd.concat(tempTr, axis=1)
				del tempTr

				pred = self.model.predict(Tr_x)
				self.fitness = fbeta_score(Tr_y, pred, beta=2)

			if self.fitnessType == "RMSE":
				pass
				# still need to implement this

			if self.fitnessType == "2FOLD-ACC":
				# *** Isto podia ser uma funcao "convert(self, X)"
				Tr_x = self.training_X
				Tr_y = self.training_Y

				tempTr = []
				for i in range(len(Tr_x.columns)):
					if self.probabilities[i] == 1:
						tempTr.append(Tr_x.iloc[:, i])
				
				Tr_x = pd.concat(tempTr, axis=1)
				del tempTr


				X1, X2, Y1, Y2 = train_test_split(Tr_x, Tr_y, train_size=0.5, random_state=42, stratify = Tr_y)

				M1 = self.create_tmp_model()
				M1.fit(X1,Y1)
				P1 = M1.predict(X2)

				M2 = self.create_tmp_model()
				M2.fit(X2,Y2)
				P2 = M2.predict(X1)

				f1 = accuracy_score(Y2, P1)
				f2 = accuracy_score(Y1, P2)
				self.fitness = (f1+f2)/2
			pass

		return self.fitness
		


	# only used for sklearn classifiers, will be changed in the future for a proper class


	def remove_cols(self, X):
		temp = []
		for i in range(len(X.columns)):
			if self.probabilities[i] == 1:
				temp.append(X.iloc[:, i])
			else:
				pass
				
		X = pd.concat(temp, axis=1)
		return X


	def getAccuracy(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		pred = self.model.predict(self.remove_cols(X))

		return accuracy_score(pred, Y)

	def getF2(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		pred = self.model.predict(self.remove_cols(X))

		return fbeta_score(Y, pred, beta=2)

	
	def getKappa(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		pred = self.model.predict(self.remove_cols(X))

		return cohen_kappa_score(pred, Y)

	def getAUC(self, X,Y,pred=None):
		'''
		Returns the individual's accuracy.
		'''
		pred = self.model.predict(self.remove_cols(X))
		
		return roc_auc_score(Y,pred)
