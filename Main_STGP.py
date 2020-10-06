import pandas
from sklearn.model_selection import train_test_split

from stgp.STGP import STGP
from sys import argv
from stgp.Constants import *
import os


# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#


def readDataset(filename, seed = 0):
	df = pandas.read_csv(filename)
	train, test = train_test_split(df, train_size = TRAIN_FRACTION, random_state = seed)
	return train, test


def run(dataset, run_id = 0):
	if VERBOSE:
		print("> Starting run:")
		print("  > ID:", run_id)
		print("  > Dataset:", dataset)
		print()

	train, test = readDataset(DATASETS_DIR+dataset, seed = run_id)
	stgp = STGP(train, test)

	accuracy = stgp.getAccuracyOverTime()
	rmse = stgp.getRmseOverTime()
	size = stgp.getSizeOverTime()
	times = stgp.getGenerationTimes()

	return ([accuracy[0],accuracy[1],rmse[0],rmse[1],size,times,str(stgp.getBestIndividual())])



def callstgp(dataset):
	toWrite=[]
	for run_id in range(RUNS):
		toWrite.append( run(dataset, run_id)  )

	openFile(OUTPUT_DIR+"stgp_"+dataset) 
	writeToFile("Attribute,Run,")
	for i in range(MAX_GENERATION):
		writeToFile(str(i)+",")
	writeToFile("\n")
		
	attributes= ["Training-Accuracy","Test-Accuracy","Training-RMSE","Test-RMSE","Size","Generation-Time","Final_Model"]
	for ai in range(len(toWrite[0])-1):
		for i in range(len(toWrite)):
			writeToFile("\n"+attributes[ai]+","+str(i)+",")
			for val in toWrite[i][ai]:
				writeToFile(str(val)+",")
		writeToFile("\n\n")
	for i in range(len(toWrite)):
		writeToFile("\n"+attributes[-1]+","+str(i)+",")
		writeToFile(str(toWrite[i][-1]))
	writeToFile("\n\n")
		
	closeFile()



if __name__ == '__main__':
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		callstgp(dataset)