from sys import argv

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-STGP
#
# Copyright ©2019-2021 J. E. Batista
#


# Operators to be used by the models
# Only these operators are available.
OPERATORS = [("+",2),("-",2),("*",2),("/",2),("log",1), ("sqrt", 1)]

# Metrics used to evaluate the model
METRICS = ["Kappa", "F2", "AUC"]

# Initial Maximum depth
MAX_DEPTH = 6

# Number of models in the GP population
POPULATION_SIZE_GP = 100

# Number of models in the GA population
POPULATION_SIZE_GA = 100

# Maximum number of iterations
MAX_GENERATION_GP = 30

# Maximum number of iterations
MAX_GENERATION_GA = 50

# Fraction of the dataset to be used as training (used by Main_M3GP_standalone.py)
TRAIN_FRACTION = 0.70

# Number of individuals to be used in the tournament
TOURNAMENT_SIZE = 5

# Number of best individuals to be automatically moved to the next generation
ELITISM_SIZE = 1

# Shuffle the dataset (used by Main_M3GP_standalone.py)
SHUFFLE = True

# Dimensions maximum depth
LIMIT_DEPTH=17

# Number of runs (used by Main_M3GP_standalone.py)
RUNS = 5

# Verbose
VERBOSE = False

# Number of CPU Threads to be used
THREADS = 10


DATASETS_DIR = "satasets/"
OUTPUT_DIR = "results/"

DATASETS = ["heart.csv"]

OUTPUT = "Classification"




if "-dsdir" in argv:
	DATASETS_DIR = argv[argv.index("-dsdir")+1]

if "-odir" in argv:
	OUTPUT_DIR = argv[argv.index("-odir")+1]

if "-d" in argv:
	DATASETS = argv[argv.index("-d")+1].split(";")

if "-runs" in argv:
	RUNS = int(argv[argv.index("-runs")+1])

if "-op" in argv:
	OPERATORS = argv[argv.index("-op")+1].split(";")

if "-md" in argv:
	MAX_DEPTH = int(argv[argv.index("-md")+1])

if "-ps" in argv:
	POPULATION_SIZE = int(argv[argv.index("-ps")+1])

if "-mg" in argv:
	MAX_GENERATION = int(argv[argv.index("-mg")+1])

if "-tf" in argv:
	TRAIN_FRACTION = float(argv[argv.index("-tf")+1])

if "-ts" in argv:
	TOURNAMENT_SIZE = int(argv[argv.index("-ts")+1])

if "-es" in argv:
	ELITISM_SIZE = int(argv[argv.index("-es")+1])

if "-dontshuffle" in argv:
	SHUFFLE = False

if "-s" in argv:
	VERBOSE = False

if "-t" in argv:
	THREADS = int(argv[argv.index("-t")+1])


