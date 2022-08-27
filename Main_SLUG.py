import pandas

from slug.GA import GA
from sys import argv
from Arguments import *
import os

from sklearn.model_selection import train_test_split

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="From version 0.21, test_size will always complement",
                        module="sklearn")


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#




def openAndSplitDatasets(which,seed):
	if VERBOSE:
		print( "> Opening: ", which )

	# Open dataset
	ds = pandas.read_csv(DATASETS_DIR+which)

	# Read header
	class_header = ds.columns[-1]

	#ds = ds.drop(columns=['ID'])
	#print(ds.columns)

	# removing first column
	ds = ds.iloc[: , 1:]

	points = {}
	for col in ds.columns[:-1]:
		points[col] = 1

	return train_test_split(ds.drop(columns=[class_header]), ds[class_header], 
		train_size=TRAIN_FRACTION, random_state=seed, 
		stratify = ds[class_header])


def run(r,dataset):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print()

	Tr_X, Te_X, Tr_Y, Te_Y = openAndSplitDatasets(dataset,r)

	# Train a model
	model = GA(population_size=POPULATION_SIZE_GA, max_generation=MAX_GENERATION_GA, elitism_size=ELITISM_SIZE, metrics=METRICS,
				GP_params={"population_size_GP":POPULATION_SIZE_GP, "max_generation_GP":MAX_GENERATION_GP, "tournament_size":TOURNAMENT_SIZE, "operators":OPERATORS, "max_depth":MAX_DEPTH, "limit_depth":LIMIT_DEPTH, "elitism_size":ELITISM_SIZE},
				threads=THREADS, verbose=VERBOSE,
				)

	model.fit(Tr_X, Tr_Y, Te_X, Te_Y)


	# Obtain training results
	acc  = model.getAccuracyOverTime()
	f2  = model.getF2OverTime()
	kappa      = model.getKappaOverTime()
	#size      = model.getSizeOverTime()
	model_str = str(model.getBestIndividual().model.getBestIndividual())
	times     = model.getGenerationTimes()
	#features = model.getBestIndividual().probabilities
	features = model.getFeaturesOverTime()
	
	tr_acc     = acc[0]
	te_acc     = acc[1]
	tr_f2     = f2[0]
	te_f2     = f2[1]
	tr_kappa    = kappa[0]
	te_kappa    = kappa[1]

	if VERBOSE:
		print("> Ending run:")
		print("  > ID:", r)
		print("  > Dataset:", dataset)
		print("  > Final model:", model_str)
		print("  > Training F2:", tr_f2[-1])
		print("  > Test Acc:", te_acc[-1])
		print("  > Test F2:", te_f2[-1])
		print("  > Training Kappa:", tr_kappa[-1])
		print("  > Test Kappa:", te_kappa[-1])
		print()

	return (tr_acc,te_acc,
			tr_f2,te_f2,
			tr_kappa,te_kappa,
			times, features,
			model_str,
			)
			

def gastgp():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		outputFilename = OUTPUT_DIR + dataset
		if not os.path.exists(outputFilename):
			results = []

			# Run the algorithm several times
			for r in range(RUNS):
				results.append(run(r,dataset))

			# Write output header
			file = open(outputFilename , "w")
			file.write("Attribute,Run,")
			for i in range(MAX_GENERATION_GA):
				file.write(str(i)+",")
			file.write("\n")
		
			attributes= ["Training-Acc","Test-Acc",
						"Training-F2","Test-F2",
						"Training-Kappa", "Test-Kappa",
						"Time",	
						"Features", "Final_Model"
						]

			# Write attributes with value over time
			for ai in range(len(attributes)-1):
				for i in range(RUNS):	
					file.write("\n"+attributes[ai]+","+str(i)+",")
					file.write( ",".join([str(val) for val in results[i][ai]]))
				file.write("\n")

			'''# Write the final features
			for i in range(len(results)):
				file.write("\n"+attributes[-2]+","+str(i)+",")
				file.write(str(results[i][-2]))
			file.write("\n")'''


			# Write the final models
			for i in range(len(results)):
				file.write("\n"+attributes[-1]+","+str(i)+",")
				file.write(results[i][-1])
			file.write("\n")

			# Write some parameters
			file.write("\n\nParameters")
			file.write("\nOperators,"+str(OPERATORS))
			file.write("\nMax Initial Depth,"+str(MAX_DEPTH))
			file.write("\nPopulation Size GA,"+str(POPULATION_SIZE_GA))
			file.write("\nPopulation Size GP,"+str(POPULATION_SIZE_GP))
			file.write("\nMax Generation GA,"+str(MAX_GENERATION_GA))
			file.write("\nMax Generation GP,"+str(MAX_GENERATION_GP))
			file.write("\nTournament Size,"+str(TOURNAMENT_SIZE))
			file.write("\nElitism Size,"+str(ELITISM_SIZE))
			file.write("\nDepth Limit,"+str(LIMIT_DEPTH))
			file.write("\nThreads,"+str(THREADS))


			file.close()
		else:
			print("Filename: " + outputFilename +" already exists.")


if __name__ == '__main__':
	gastgp()
