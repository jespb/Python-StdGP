import pandas

from stgp.STGP import STGP
from sys import argv
from stgp.Constants import *
import os

import time

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019 J. E. Batista
#

timestamp = time.strftime("%Y%m%d_%H%M")

def callstgp():
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		pass

	for dataset in DATASETS:
		openFile(OUTPUT_DIR+"tmp_stgp_"+timestamp + "_"+dataset)
		writeToFile(dataset+"\n")
		toWrite=[]
		for i in range(RUNS):
			print(i,"# run with the", dataset,"dataset")
			p = pandas.read_csv(DATASETS_DIR+dataset)
			stgp = STGP(p)

			writeToFile(",")
			for i in range(MAX_GENERATION):
				writeToFile(str(i)+",")
			
			accuracy = stgp.getAccuracyOverTime()
			rmse = stgp.getRmseOverTime()
			size = stgp.getSizeOverTime()
			toWrite.append([accuracy[0],accuracy[1],rmse[0],rmse[1],size,str(stgp.getBestIndividual())])
			
			writeToFile("\nTraining-Accuracy,")
			for val in accuracy[0]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTest-Accuracy,")
			for val in accuracy[1]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTraining-RMSE,")
			for val in rmse[0]:
				writeToFile(str(val)+",")
			
			writeToFile("\nTest-RMSE,")
			for val in rmse[1]:
				writeToFile(str(val)+",")

			writeToFile("\nSize,")
			for val in size:
				writeToFile(str(val)+",")

			writeToFile("\n"+str(stgp.getBestIndividual())+"\n")
		
		closeFile()

		openFile(OUTPUT_DIR+"stgp_"+timestamp + "_"+dataset) 
		writeToFile("Attribute,Run,")
		for i in range(MAX_GENERATION):
			writeToFile(str(i)+",")
		writeToFile("\n")
		
		attributes= ["Training-Accuracy","Test-Accuracy","Training-RMSE","Test-RMSE","Size","Dimensions","Final_Model"]
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
		os.remove(OUTPUT_DIR+"tmp_stgp_"+timestamp + "_"+dataset)

callstgp()