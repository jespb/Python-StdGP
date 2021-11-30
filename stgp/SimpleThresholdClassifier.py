
# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#


class SimpleThresholdClassifier:

	threshold = None
	isPositive1 = True

	def __init__(self, threshold = 0):
		self.threshold = threshold

	def fit(self,X=None,Y=None):
		pass


	def predict(self, X):	
		"""
		Receives X, a 1-D array of real values
		Return a list of predictions based on the value
		"""	
		if self.isPositive1:
			predictions = [ 1 if value > self.threshold else 0 for value in X]
		else:
			predictions = [ 1 if value < self.threshold else 0 for value in X]
		return predictions

	def invertPredictions(self):
		self.isPositive1 = not self.isPositive1
