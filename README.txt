This is a, easy-to-use, scikit-learn inspired version of the SLUG algorithm.


By using this file, you are agreeing to this product's EULA
This product can be obtained in https://github.com/NMVRodrigues/SLUG
Copyright ©2021-2023 Nuno M. Rodrigues

Project forked from https://github.com/jespb/Python-StdGP


This file contains information about the command and flags used in the stand-alone version of this implementation and an explanation on how to import, use and edit this implementation.


This implementation of SLUG can be used in a stand-alone fashion using the arguments specified in the Arguments.py file

$ python Main_SLUG.py
	
	
How to import this implementation to your project:
	- Download this repository;
	- Copy the "slug/" directory to your project directory;
	- import the GA class using "from slug.GA import GA".


Arguments for GA():
	operators			-> Operators used by the individual (default: ["+","-","*","/"] )
	max_depth			-> Max initial depths of the individuals (default: 6)
	population_size_GA		-> Population size for the GA (default: 100)
	population_size_GP		-> Population size for the GP (default: 100)
	max_generation_GA		-> Maximum number of generations for the GA (default: 30)
	max_generation_GP		-> Maximum number of generations for the GP (default: 50)
	tournament_size		-> Tournament size (default: 5)
	elitism_size		-> Elitism selection size (default: 1)
	limit_depth			-> Maximum individual depth (default: 17)
	threads 			-> Number of CPU threads to be used (default: 1)

Arguments for model.fit():
	Tr_X 				-> Training samples
	Tr_Y 				-> Training labels
	Te_X 				-> Test samples, used in the standalone version (default: None)
	Te_Y 				-> Test labels, used in the standalone version (default: None)


Useful methods:
	$ model = GA()			-> starts the model;
	$ model.fit(X, Y)			-> fits the model to the dataset;
	$ model.predict(dataset)    -> Returns a list with the prediction of the given dataset.




How to edit this implementation:
	Fitness Function ( slug.Individual_GA ):
		- Change the getFitness() method to use your own fitness function;
		- This implementation assumes that a higher fitness is always better. To change this, edit the __gt__ method in this class;
		- You may use the getTrainingPredictions() and getTrainingSet() to obtain the models prediction and the training set;
		- You can also explore the behind the standard fitness function;
		- Warning: GA evaluates every model in every run, as such, I do not recommend complex fitness functions. You should invest in fast evaluation methods to train a population.

	Classification method ( slug.Individual_GA ):
		- Change the create() method to use your own classifier;
		- Assuming it is a scikit-learn implementation, you may only need to change the first few lines of this method;
		- Warning: GA evaluates every model in every run, as such, I do not recommend complex classification model. You should invest in fast classification methods to train a population and the use a more complex method (if you wish) on the final model.


Reference:
    Poli, R., Langdon, W.B., McPhee, N.F.: A Field Guide to Genetic Programming. Lulu Enterprises, UK Ltd (2008)