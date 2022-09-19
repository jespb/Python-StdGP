# Imports utility libraries 
import datetime
import os
from socketserver import ThreadingUnixStreamServer
from .Individual import Individual
import pandas as pd
# Imports PyTorch
from sklearn.metrics import accuracy_score
import torch
# Imports metaheuristics 
from gpol.algorithms.genetic_algorithm import GSGP
from gpol.algorithms.local_search import HillClimbing, SimulatedAnnealing
# Imports metaheuristics 
from gpol.algorithms.random_search import RandomSearch
# Imports operators
# Imports operators
from gpol.operators.initializers import grow, prm_full, rhh
from gpol.operators.selectors import prm_tournament
from gpol.operators.variators import (prm_efficient_gs_mtn,
                                      prm_efficient_gs_xo)
# Imports problems
from gpol.problems.inductive_programming import SML, SMLGS
from gpol.utils.datasets import load_boston
from gpol.utils.inductive_programming import (_execute_tree, _get_tree_depth,
                                              function_map,
                                              prm_reconstruct_tree)
from gpol.utils.utils import train_test_split
# Imports problem
from torch.utils.data import DataLoader, TensorDataset


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1))

class GSGP_Indiv:
    population = None

    operators = None
    max_depth = None
    population_size = None
    max_generation = None
    tournament_size = None
    elitism_size = None
    limit_depth =None
    threads = None
    verbose = None


    def checkIfTrained(self):
        if self.population == None:
            raise Exception("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")


    def __init__(self, operators=["+","-","*","/"], max_depth = 6, population_size = 100, 
        max_generation = 30, tournament_size = 5, elitism_size = 1, limit_depth = 8, 
        threads=1, verbose = False):

        if sum( [0 if op in ["+","-","*","/"] else 0 for op in operators ] ) > 0:
            print( "[Warning] Some of the following operators may not be supported:", operators)
        self.operators = operators
        self.max_depth = max_depth
        self.population_size = population_size
        self.max_generation = max_generation
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.limit_depth = limit_depth
        self.threads = max(1, threads)
        self.verbose = verbose
        pass

    def __str__(self):
        self.checkIfTrained()
        return str(self.getBestIndividual())



    def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
        
        if self.verbose:
            print("Training a model with the following parameters: ", end="")
            print("{Operators : "+str(self.operators)+"}, ", end="")
            print("{Max Initial Depth : "+str(self.max_depth)+"}, ", end="")
            print("{Population Size : "+str(self.population_size)+"}, ", end="")
            print("{Max Generation : "+str(self.max_generation)+"}, ", end="")
            print("{Tournament Size : "+str(self.tournament_size)+"}, ", end="")
            print("{Elitism Size : "+str(self.elitism_size)+"}, ", end="")
            print("{Depth Limit : "+str(self.limit_depth)+"}, ", end="")
            print("{Threads : "+str(self.threads)+"}, ", end="")
        
        # Convert dataframes to tensors
        Tr_X_tensor = torch.tensor(Tr_X.values)
        Tr_Y_tensor = torch.tensor(Tr_Y.values)
        if Te_X is not None:
            Te_X_tensor = torch.tensor(Te_X.values)
            Te_Y_tensor = torch.tensor(Te_Y.values)

            X = torch.concat((Tr_X_tensor, Te_X_tensor))
            y = torch.concat((Tr_Y_tensor, Te_Y_tensor))
            p_test = len(Tr_X_tensor) / len(X)
        else:
            X = Tr_X_tensor
            y = Tr_Y_tensor
            p_test = 0.3

        # Defines parameters for the data usage
        batch_size, shuffle = 50, False
        device, seed  = 'cpu', 0  # 'cuda' if torch.cuda.is_available() else 'cpu', 0            # Performs train/test split
        
        train_indices, test_indices = train_test_split(X=X, y=y, p_test=p_test, shuffle=shuffle, indices_only=True, seed=seed)
        # Characterizes the program elements: function and constant sets
        f_set = [function_map["add"], function_map["sub"], function_map["mul"], function_map["div"]]
        c_set = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)
        # Creates the search space
        sspace = {"n_dims": X.shape[1], "function_set": f_set, "p_constants": 0.1, "constant_set": c_set, "max_init_depth": 5}
        # Creates problem's instance
        pi = SMLGS(sspace=sspace, ffunction=rmse, X=X, y=y, train_indices=train_indices, test_indices=test_indices,
                batch_size=100, min_=True)

        # Defines population's size
        pop_size = 100 * 3
        # Creates single trees' initializer for the GSOs
        sp_init = prm_full(sspace)  
        # Generates GSM's steps
        to, by = 5.0, 0.25  
        ms = torch.arange(by, to + by, by, device=device)
        # Defines selection's pressure and mutation's probability
        pars = {"pop_size": pop_size, "initializer": rhh, "selector": prm_tournament(pressure=0.1),
                "mutator": prm_efficient_gs_mtn(X, sp_init, ms), "crossover": prm_efficient_gs_xo(X, sp_init),
                "p_m": 0.3, "p_c": 0.7, "elitism": True, "reproduction": False}

        # Creates the experiment's label
        experiment_label = "SMLGS"  # SML approached from the perspective of Inductive Programming using GSGP
        time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
                str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
        # Creates general path
        path = os.path.join(os.getcwd(), experiment_label, time_id)
        # Defines a connection string to store random trees
        path_rts = os.path.join(path, "reconstruct", "rts")
        if not os.path.exists(path_rts):
            os.makedirs(path_rts)
        # Defines a connection string to store the initial population
        path_init_pop = os.path.join(path, "reconstruct", "init_pop")
        if not os.path.exists(path_init_pop):
            os.makedirs(path_init_pop)
        # Creates a connection string towards the history's file
        path_hist = os.path.join(path, "reconstruct", experiment_label + "_seed_" + str(seed) + "_history.csv")  
        
        #n_iter = self.max_generation
        n_iter = 10

        isa = GSGP(pi=pi, path_init_pop=path_init_pop, path_rts=path_rts, seed=seed, device=device, **pars)
        isa.solve(n_iter=n_iter, tol=0.1, n_iter_tol=n_iter, test_elite=True, verbose=0, log=0)
        isa.write_history(path_hist)
        #print("Algorithm: {}".format(isa.__name__))
        #print("Best solution's fitness: {:.3f}".format(isa.best_sol.fit))
        #print("Best solution's test fitness: {:.3f}".format(isa.best_sol.test_fit)) 
        history = pd.read_csv(os.path.join(path_hist), index_col=0)
        # Creates a reconstruction function
        reconstructor = prm_reconstruct_tree(history, path_init_pop, path_rts, device)
        # Chooses the most fit individual to reconstruct
        start_idx = history["Fitness"].idxmin()
        #print("Starting index (chosen individual):", start_idx)
        #print("Individual's info:\n", history.loc[start_idx])
        # Reconstructs the individual
        ind = reconstructor(start_idx)
        #print("Automatically reconstructed individual's representation:\n", ind[0:30])

        self.history = history
        list_of_unique_numbers = []
        unique_numbers = set(ind)
        for number in unique_numbers:
            if type(number) == int:
                list_of_unique_numbers.append(number)

        self.best_tree = ind
        self.population = isa.pop
        self.best_fit = accuracy_score(torch.round(_execute_tree(self.best_tree, Tr_X_tensor[test_indices])), Tr_Y_tensor[test_indices] )
        #print(f"Fitness {self.best_fit}, {len(list_of_unique_numbers)}/{X.shape[1]} features {len(list_of_unique_numbers)/X.shape[1] * 100}%")

    def predict(self, dataset):
        '''
        Returns the predictions for the samples in a dataset.
        '''
        self.checkIfTrained()
        data_tensor = torch.tensor(dataset.values)

        return torch.round(torch.clamp(_execute_tree(self.best_tree, data_tensor), 0, 1))

    def getBestIndividual(self):
        '''
        Returns the final M3GP model.
        '''
        self.checkIfTrained()
        return self.best_tree

    def getAccuracyOverTime(self):
        '''
        Returns the training and test accuracy of the best model in each generation.
        '''
        pass

    def getRMSEOverTime(self):
        '''
        Returns the training and test accuracy of the best model in each generation.
        '''
        pass


        pass
    def getF2OverTime(self):
        '''
        Returns the training and test accuracy of the best model in each generation.
        '''

        pass

    def getKappaOverTime(self):
        '''
        Returns the training and test accuracy of the best model in each generation.
        '''
        pass

    def getAUCOverTime(self):
        '''
        Returns the training and test AUC of the best model in each generation.
        '''
        pass

    def getSizeOverTime(self):
        '''
        Returns the size and number of dimensions of the best model in each generation.
        '''
        pass

    def getGenerationTimes(self):
        '''
        Returns the time spent in each generation.
        '''
        pass
