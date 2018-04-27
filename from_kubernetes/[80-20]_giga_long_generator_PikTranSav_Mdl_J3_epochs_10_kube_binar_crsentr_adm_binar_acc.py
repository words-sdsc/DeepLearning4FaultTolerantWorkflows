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
path = "."

# Declare number of GPUs to make visible to your script
pickgpus = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]=pickgpus

startup = False
noofiterations = 20
model = []

def create_modelJ3(input_dim, input_length, output_dim):
    print ('Creating model J3 (Pick Train Save)...')
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(100, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(100, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(100, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(100, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(output_dim, activation='sigmoid'))
    ###
    ###
    ###
    print('Initiating parallel GPU model')
    parallel_model = multi_gpu_model(model, gpus=1+pickgpus.count(","))
    print ('Compiling...')
    parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    return parallel_model

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
    ######### Testing ##########
    ######### Testing ##########

    # Load each file, predict, and coagulate the Y_predict, Y_test
    Y_true = []
    Y_pred = []

    for k in range(len(X_test_files)):
        # Start with index 0,1,2,3,4,...

        # Load X_test:
        print("Loading X_test file # ", k, X_test_files[k] )
        with open(X_test_files[k], 'rb') as handle:
            X_test = pickle.load(handle)

        # Load Y_test:
        print("Loading Y_test file # ", k, Y_test_files[k] )
        with open(Y_test_files[k], 'rb') as handle:
            Y_test = pickle.load(handle)

        # Predict:
        y_pred_this_file      = model.predict(X_test, batch_size = 4*1024*2)

        # Concatenate the results:
        
        Y_true.extend([ xj[0] for xj in Y_test ])
        Y_pred.extend([ np.rint(jj[0]) for jj in y_pred_this_file ])
        
        del X_test
        del Y_test
        gc.collect() 

    ########### Print Test Results to disk
    ########### Print Test Results to disk
    ########### Print Test Results to disk
    ########### Print Test Results to disk

    print('********* Test Results Start ************')
    print('[80-20]_giga_long_generator_PikTranSav_Mdl_J3_epochs_10_kube_binar_crsentr_adm_binar_acc.h5')
    print("Overall accuracy on test set: ", np.mean(np.equal(Y_true, Y_pred)))
    print("Calculating Score: ")

    precision, recall, fscore, support = score(Y_true, Y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print("Confusion matrix:")
    print(pd.crosstab(pd.Series(Y_true), pd.Series(Y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
    print('*********Test Results End**************')
    
    
# Create list of training file names

X_train_files = [f for f in listdir(path) if isfile(join(path, f))]
X_train_files = [x for x in X_train_files if '_long__file_' in x and 'X_final_' in x ]
X_train_files.sort()
print('# of X_train files detected: ', len(X_train_files))

Y_train_files = [f for f in listdir(path) if isfile(join(path, f))]
Y_train_files = [x for x in Y_train_files if '_long__file_' in x and 'Y_final_' in x ]
Y_train_files.sort()
print('# of Y_train files detected: ', len(Y_train_files))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
## Training data generator
def train_generator():
    while True:
        for j in range(len(X_train_files)):
            # Load X_final
            print("Loading X_train file # ", j, X_train_files[j] )
            with open(X_train_files[j], 'rb') as handle:
                X_final = pickle.load(handle)
            print('Loaded X_final shape: ', X_final.shape)
            # Load Y_final
            print("Loading Y_train file # ", j, Y_train_files[j] )
            with open(Y_train_files[j], 'rb') as handle:
                Y_final = pickle.load(handle)
            print('Loaded Y_final shape: ', Y_final.shape)
            
            howmany = len(X_final)
            # break into 8 pieces
            for bat in batch(range(howmany), max(2, int(howmany/20)) ):
                yield (X_final[bat], Y_final[bat])

# Define dimensionality
input_length    = 1    # X_final.shape[1]
input_dim       = 3541 # X_final.shape[2]
output_dim      = 1
#
totalbatches = 21*40+8   # =number of samples of your dataset / average batch size
# 40 files will be broken into 5 pieces                (21*40 pieces)
# 8 files will be returned as it is with size = 2      (8 pieces)
# Total no of batched = 40*26+8*1

model   = create_modelJ3(input_dim, input_length, output_dim)
print('Fit the Model...Calling fit_generator ')

history=model.fit_generator(train_generator(),steps_per_epoch=totalbatches,epochs=10,verbose=1,use_multiprocessing=True, workers=1, max_queue_size=25)

print("Saving Model weights...")

model.save_weights('[80-20]_giga_long_generator_PikTranSav_Mdl_J3_epochs_10_kube_binar_crsentr_adm_binar_acc.h5')
########################################################
########################################################
# Print test results after going through each file once
print('End of Outer Iteration Testing (gone thru each file once) everyiter= ', everyiter)
printTestResults(model)
