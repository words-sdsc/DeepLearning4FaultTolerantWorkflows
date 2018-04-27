# This is a tester
# Change the function g - that creates the model
# Call g to create model
# Load weights
# Call function h - to test and print test results

from keras.utils import multi_gpu_model
import os
import gc
import pandas as pd
import numpy as np
import sys
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
import pickle 
from sklearn.metrics import precision_recall_fscore_support as score
from os import listdir
from os.path import isfile, join

# FIX 1 of 3 : set the following vars
path                 = "/oasis/scratch/comet/a1singh/temp_project/tsteps2" #path to dir where test pickles reside and h5 file is
modelfingerprint     = "comet_train_[80-20]_giga_long_multithreadedgenerator_PikTranSav_Mdl_[KTAPER_tsteps2]_epochs_8_kube_binar_crsentr_adm_binar_acc.py" # name of py file
modelweightsfilename = "comet_train_[80-20]_giga_long_multithreadedgenerator_PikTranSav_Mdl_[KTAPER_tsteps2]_epochs_8_kube_binar_crsentr_adm_binar_acc.h5" # name of h5 file

# FIX 2 of 3: update the input_length as per timesteps
input_length    = 2 # 1 or 2                        # X_final.shape[1]
input_dim       = 3541                              # X_final.shape[2]
output_dim      = 1

#pickgpus = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]=pickgpus

def create_model(input_dim, input_length, output_dim):
    print ('Creating model...')
    model = Sequential()
    model.add(LSTM(input_dim, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(int(input_dim/2) , input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(int(input_dim/4) , input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(int(input_dim/8) ))
    model.add(Dense(output_dim, activation='sigmoid'))
    ###
    ###
    print('Initiating parallel GPU model')
    parallel_model = multi_gpu_model(model, gpus=1+1) #pickgpus.count(","))
    print ('Compiling...')
    parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    return parallel_model

##################### stable below this ###########################
###################################################################

def printTestResults(model):
    print('Tester function called...using all test files')
    # Create list of test file names
    X_test_files = [f for f in listdir(path) if isfile(join(path, f))]
    X_test_files = [x for x in X_test_files if '_long__file_' in x and 'X_test_' in x ]
    X_test_files.sort()
    print('# of X_test files detected: ', len(X_test_files))

    Y_test_files = [f for f in listdir(path) if isfile(join(path, f))]
    Y_test_files = [x for x in Y_test_files if '_long__file_' in x and 'Y_test_' in x ]
    Y_test_files.sort()
    print('# of Y_test files detected: ', len(Y_test_files))

    ######### Testing ##########
    ######### Testing ##########

    # Load each file, predict, and coagulate the Y_predict, Y_test
    Y_true = []
    Y_pred = []

    for k in range(len(X_test_files)):
        # Start with index 0,1,2,3,4,...

        # Load X_test:
        print("Loading X_test file # ", k, X_test_files[k] )
        with open( join(path, X_test_files[k]) , 'rb' ) as handle:
            X_test = pickle.load(handle)

        # Load Y_test:
        print("Loading Y_test file # ", k, Y_test_files[k] )
        with open( join(path, Y_test_files[k]), 'rb' ) as handle:
            Y_test = pickle.load(handle)

        # Predict:
        y_pred_this_file      = model.predict(X_test, batch_size = 4*1024*2)

        # Concatenate the results:
        Y_true.extend([ xj[0] for xj in Y_test ])
        Y_pred.extend([ np.rint(jj[0]) for jj in y_pred_this_file ])
        
        del X_test
        del Y_test
        print('####')
        gc.collect() 

    ########### Print Test Results to disk
    ########### Print Test Results to disk
    ########### Print Test Results to disk
    ########### Print Test Results to disk

    print('********* Test Results Start **************')

    print('********* Model Fingerprint >> ************')
    for i in range(5):
        print(modelfingerprint)
    print('********* Model Fingerprint << ************')

    print('Len of Y_true : ', len(Y_true))
    print('Len of Y_pred : ', len(Y_pred))

    print("Overall accuracy on test set: ", np.mean(np.equal(Y_true, Y_pred)))
    
    print("Calculating Score: ")
    precision, recall, fscore, support = score(Y_true, Y_pred)
    print('precision    : {}'.format(precision))
    print('recall       : {}'.format(recall))
    print('fscore       : {}'.format(fscore))
    print('support      : {}'.format(support))

    print("Confusion matrix:")
    print(pd.crosstab(pd.Series(Y_true), pd.Series(Y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
    print('*********Test Results End**************')

# STEP 1
model   = create_model(input_dim, input_length, output_dim)

# STEP 2
print('Loading Model weights before Testing')
model.load_weights( join(path, modelweightsfilename) )

print('Calling printTestResults() for testing the model:')
printTestResults(model)

print('Test Program has reach the last line of execution')
