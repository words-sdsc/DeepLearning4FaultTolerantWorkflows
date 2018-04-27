import os
import gc
import pandas as pd
import numpy as np
import sys
import pickle 
from sklearn.metrics import precision_recall_fscore_support as score
from os import listdir
from os.path import isfile, join
path = "."

def split_test_XY():
    # Create list of training file names

    # Create list of test file names
    X_test_files = [f for f in listdir(path) if isfile(join(path, f))]
    X_test_files = [x for x in X_test_files if '_long__file_' in x and 'X_test_' in x ]
    X_test_files.sort()

    Y_test_files = [f for f in listdir(path) if isfile(join(path, f))]
    Y_test_files = [x for x in Y_test_files if '_long__file_' in x and 'Y_test_' in x ]
    Y_test_files.sort()
    
    ######## Notice the laziness ! Danger
    X_train_files = X_test_files
    Y_train_files = Y_test_files
    ######## Notice the laziness ! Danger

    for j in range(len(X_train_files)):
        print('Test X File # ', j)

        # Load X_final
        print("Loading X test file # %d", j)
        with open(X_train_files[j], 'rb') as handle:
            X_final = pickle.load(handle)
        print('Loaded X test shape: ', X_final.shape)

        A, B = np.array_split(X_final, 2)

        #Dump 1
        with open('Part1_' + X_train_files[j], 'wb') as handle:
            pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del A

        #Dump 2
        with open('Part2_' + X_train_files[j], 'wb') as handle:
            pickle.dump(B, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del B
        del X_final
        gc.collect()

        # Load Y_test
        
        print("Loading Y test file # ", j)
        with open(Y_train_files[j], 'rb') as handle:
            Y_final = pickle.load(handle)
        print('Loaded Y test shape: ', Y_final.shape)

        AA, BB = np.array_split(Y_final, 2)

        #Dump Y1
        with open('Part1_' + Y_train_files[j], 'wb') as handle:
            pickle.dump(AA, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del AA

        #Dump Y2
        with open('Part2_' + Y_train_files[j], 'wb') as handle:
            pickle.dump(BB, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del BB
        del Y_final
        gc.collect()
        ############
        
        
def split_train_XY():
    # Create list of training file names

    X_train_files = [f for f in listdir(path) if isfile(join(path, f))]
    X_train_files = [x for x in X_train_files if '_long__file_' in x and 'X_final_' in x ]
    X_train_files.sort()

    Y_train_files = [f for f in listdir(path) if isfile(join(path, f))]
    Y_train_files = [x for x in Y_train_files if '_long__file_' in x and 'Y_final_' in x ]
    Y_train_files.sort()

    for j in range(len(X_train_files)):
        print('Training File # ', j)

        # Load X_final
        print("Loading X_final file # %d", j)
        with open(X_train_files[j], 'rb') as handle:
            X_final = pickle.load(handle)
        print('Loaded X_final shape: ', X_final.shape)

        A, B = np.array_split(X_final, 2)

        #Dump 1
        with open('Part1_' + X_train_files[j], 'wb') as handle:
            pickle.dump(A, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del A

        #Dump 2
        with open('Part2_' + X_train_files[j], 'wb') as handle:
            pickle.dump(B, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del B
        del X_final
        gc.collect()

        # Load Y_final and Split Labels
        
        print("Loading Y_final file # ", j)
        with open(Y_train_files[j], 'rb') as handle:
            Y_final = pickle.load(handle)
        print('Loaded Y_final shape: ', Y_final.shape)

        AA, BB = np.array_split(Y_final, 2)

        #Dump Y1
        with open('Part1_' + Y_train_files[j], 'wb') as handle:
            pickle.dump(AA, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del AA

        #Dump Y2
        with open('Part2_' + Y_train_files[j], 'wb') as handle:
            pickle.dump(BB, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del BB
        del Y_final
        gc.collect()
        
print("Splitting test files first")
split_test_XY()
print("Now splitting train files")
split_train_XY()
