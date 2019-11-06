import pandas

from stgp.STGP import STGP
from sys import argv
from stgp.Constants import *

import time

#
# $ python Main_STGP.py [-dsdir dir] [-d datasets] [-r]
#
# [-dsdir dir] 
# 	- States the dataset directory. 
# 	- By default "datasets/" is used 
# 	- Use "-dsdir ./" for the root directory
# [-d datasets] 
# 	- this flag expects a set of csv dataset names separated by ";" (e.g., a.csv;b.csv)
# [-r] 
# 	- States the this is a regression problem. 
# 	- By default the STGP tries to classify samples as 0 or 1
#

timestamp = time.strftime("%Y%m%d_%H%M")

ds_dir = "datasets/"

datasets = ["heart.csv"] #example dataset
output = "Classification"

if "-dsdir" in argv:
	ds_dir = argv[argv.index("-dsdir")+1]
if "-d" in argv:
	datasets = argv[argv.index("-d")+1].split(";")
if "-r" in argv:
	output = "Regression"
if "-runs" in argv:
	RUNS = int(argv[argv.index("-runs")+1])

def callstgp():
	#for dataset in datasets:
	#	openFile("out_"+dataset)
	#	writeToFile(dataset+"\n")
	#	for i in range(RUNS):
	#		print(i,"# run with the ", dataset,"dataset")
	#		p = pandas.read_csv(ds_dir+dataset)
	#		stgp = STGP(p,output)
	#		writeToFile(str(i)+",")
	#		writeToFile(str(stgp.getCurrentGeneration())+",")
	#		writeToFile(str(stgp.getTrainingAccuracy())+",")
	#		writeToFile(str(stgp.getTestAccuracy())+",")
	#		writeToFile(str(stgp.getTrainingRMSE())+",")
	#		writeToFile(str(stgp.getTestRMSE())+",")
	#	closeFile()

	for dataset in datasets:#["trio_brasil.csv","trio_congo.csv","trio_mocambique.csv","trio_combo.csv"]:#["mcd3.csv","mcd10.csv","brasil.csv","movl.csv","heart.csv","vowel.csv","wav.csv","yeast.csv","seg.csv"]:
		openFile("stgp_"+timestamp + "_"+dataset)
		writeToFile(dataset+"\n")
		toWrite=[]
		for i in range(RUNS):
			print(i,"# run with the", dataset,"dataset")
			p = pandas.read_csv(ds_dir+dataset)
			stgp = STGP(p,output)

			writeToFile(",")
			for i in range(MAX_GENERATION):
				writeToFile(str(i)+",")
			
			fitness = stgp.getFitnessOverTime()
			size = stgp.getSizeOverTime()
			toWrite.append([fitness[0],fitness[1],size,str(stgp.getBestIndividual())])
			
			writeToFile("\nTraining,")
			for val in fitness[0]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTest,")
			for val in fitness[1]:
				writeToFile(str(val)+",")

			writeToFile("\nSize,")
			for val in fitness[0]:
				writeToFile(str(val)+",")

			writeToFile("\n"+str(stgp.getBestIndividual())+"\n")
		
		closeFile()

		openFile("stgp_"+timestamp + "_"+dataset) 
		writeToFile("Attribute,Run,")
		for i in range(MAX_GENERATION):
			writeToFile(str(i)+",")
		writeToFile("\n")
		
		attributes= ["Training","Test","Size","Dimensions","Final_Model"]
		for ai in range(len(toWrite[0])-1):
			for i in range(len(toWrite)):
				writeToFile("\n"+attributes[ai]+","+str(i)+",")
				for val in toWrite[i][ai]:
					writeToFile(str(val)+",")
				#writeToFile(",".join(toWrite[i][ai]))
			writeToFile("\n\n")
		for i in range(len(toWrite)):
			writeToFile("\n"+attributes[-1]+","+str(i)+",")
			writeToFile(str(toWrite[i][-1]))
		writeToFile("\n\n")

		
		closeFile()

callstgp()