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
import shutil

path = "/oasis/scratch/comet/a1singh/temp_project/"

# Declare number of GPUs to make visible to your script
pickgpus = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"]=pickgpus

startup         = False
noofiterations  = 50
model           = []

def create_modelJ1700(input_dim, input_length, output_dim):
    print ('Creating model J1700 (Pick Train Save)...')
    
    model = Sequential()
    model.add(LSTM(1700, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(1700, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(1700, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(1700, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(1700, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(1700))
    model.add(Dense(output_dim, activation='sigmoid'))
    ###
    ###
    ###
    print('Initiating parallel GPU model')
    parallel_model = multi_gpu_model(model, gpus=1+pickgpus.count(","))
    print ('Compiling...')
    parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    return parallel_model  
    
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
    
##$$$$$$$$
import threading
import multiprocessing
import time
lock    = threading.Lock()
maxproc = 4

## global flags
flag            = False
programendflag  = False
processes       = []

## global flags
queue         = multiprocessing.Queue(maxsize=200)
threadcounter = multiprocessing.Queue(maxsize=1)
threadcounter.put(0)

def getj():
    global threadcounter

    lock.acquire(blocking=True)
    try:
        retval        = threadcounter.get()
        print('getj() : got the lock & threadcounter=', retval)
        newthreadcounter = (1+retval)%48  #there are 48 training files
        print('getj() : updated threadcounter=', newthreadcounter)
        threadcounter.put(newthreadcounter)
        return X_train_files[retval], Y_train_files[retval], retval
    finally:
        lock.release()
# define producer (putting items into queue)

partitions = 20

def producer():
    global maxproc
    global flag
    global queue
    global processes

    try:
        # Get the value of j for j in range(len(X_train_files)):
        xfile, yfile, j = getj()
        xfile = path+xfile
        yfile = path+yfile

        # Load X_final
        print("\n######## producer()  : Loading X_train file #", j, xfile )
        with open(xfile, 'rb') as handle:
            X_final = pickle.load(handle)
        print('######## producer()  : Loaded X_train file #', j, X_final.shape, xfile)
        # Load Y_final
        print("######## producer()  : Loading Y_train file #", j, yfile)
        with open(yfile, 'rb') as handle:
            Y_final = pickle.load(handle)
        print('######## producer()  : Loaded Y_train file #', j, Y_final.shape, yfile)
        
        howmany = len(X_final)
        #
        for bat in batch(range(howmany), max(2, int(howmany/partitions)) ):
            queue.put([ X_final[bat], Y_final[bat] ])

        # this flag allows the first time filled signal to pass to entire program
        print('######## producer()  : Setting flag = True; it was : ', flag)
        flag = True

    except Exception as e: 
        print("\n######## producer()  : Exception caught in producer() ")
        print(e)
    ### end of producer

def start_process(): #inner function
    global maxproc
    global flag
    global queue
    global processes

    for i in range(len(processes), maxproc): #never go avove maxproc
        thread = multiprocessing.Process(target=producer)
        time.sleep(0.02)
        thread.start()
        processes.append(thread)

# Define a generator function
def factory():
    global maxproc
    global flag
    global queue
    global processes

    """ Use multiprocessing to generate batches in parallel. """
    print('factory called')
    try:  
        # processes = []  < moved to main
        # <<<<<<<<<<<<<<<<< inner functions moves out
        # Check and Fill the Queue
        # 
        while True: #this thread will remain alive
            processes   = [p for p in processes if p.is_alive()]
            lim         = int(0.5*((partitions+1)*maxproc))
            if queue.qsize() < lim and len(processes) < maxproc:
                print('\n######## factory():Queue size<', lim, ' (', queue.qsize(),') launch producers, right now we have', len(processes))
                start_process()
                time.sleep(60)

            time.sleep(1)
            if(programendflag == True):
                print('\n######## factory() : program end signal...break')
                break
    except:
        print("Finishing")
        for th in processes:
            th.terminate()
        queue.close()
        raise
    finally:
        if (programendflag == True):
            for th in processes:
                th.terminate()
            queue.close()


##$$$$$$$$
from threading import Thread

# Thread that keeps filling the queue
thread_factory = multiprocessing.Process(target=factory, args=() )
time.sleep(0.02)
thread_factory.start()

while(queue.qsize() < 1): #first time filling of Q
    # wait for first fill
    print('main() : waiting for queue to fill first time ... cleaning & sleeping 20 secs')
    print('main() : status of global variable flag: ', flag)
    print('main() : length of var processes: ', len(processes))
    gc.collect()
    time.sleep(20)

## Training data generator
def train_generator():
    global maxproc
    global flag
    global queue
    global processes
    #
    while True:
        try:
            while(queue.qsize() < 2):
                print('\n ^^^^^^^^ train_generator() : Q is empty cleaning & sleeping 30 secs...factory shud be filling it ...')
                print('^^^^^^^^ train_generator() : status of global variable flag: ', flag)
                print('^^^^^^^^ train_generator() : length of var processes: ', len(processes))
                print('^^^^^^^^ train_generator() : length of queue: ', queue.qsize())
                gc.collect()
                time.sleep(30)
            #
            #
            
            pick = queue.get()
            if(len(pick[0]) > 0):
                yield pick[0], pick[1]
        except Exception as e: 
            print("train_generator: Exception caught when performing queue.get()")
            print(e)
            print('train_generator: Cleaning & Sleeping for 30 secs... someone should be filling the queue ...')
            gc.collect()
            time.sleep(30)

        """
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
            for bat in batch(range(howmany), max(2, int(howmany/10)) ):
                yield (X_final[bat], Y_final[bat])
                """

# Define dimensionality
input_length    = 1    # X_final.shape[1]
input_dim       = 3541 # X_final.shape[2]
output_dim      = 1
#
totalbatches = (1+partitions)*40+8  # =number of samples of your dataset / average batch size
# 40 files will be broken into 'partitions+1' # of pieces                (partitions*40 pieces)
# 8 files will be returned as it is with size = 2      (8 pieces)
# Total no of batched = 40*(partitions+1)+8*1

model   = create_modelJ1700(input_dim, input_length, output_dim)
print('Fit the Model...Calling fit_generator ')

history=model.fit_generator(train_generator(),steps_per_epoch=totalbatches,epochs=noofiterations,verbose=1,use_multiprocessing=True, workers=1, max_queue_size=2)
print("Saving Model weights...")

model.save_weights('comet_train_[80-20]_giga_long_multithreadedgenerator_PikTranSav_Mdl_J1700_epochs_20_kube_binar_crsentr_adm_binar_acc.h5')

model.save_weights('/home/a1singh/lstm_giga_data/results/comet_train_[80-20]_giga_long_multithreadedgenerator_PikTranSav_Mdl_J1700_epochs_20_kube_binar_crsentr_adm_binar_acc.h5')

print('Weights Saved')

########################################################
########################################################
# Print test results after going through each file once
print('End of Training: Use different script for Printing test results')
programendflag = True
print('Program has reach the last line of execution')

print('Start copy of files...')
shutil.copy2('./*.h5',  '/home/a1singh/lstm_giga_data/results')
shutil.copy2('./*.log', '/home/a1singh/lstm_giga_data/results')
shutil.copy2('./*.py',  '/home/a1singh/lstm_giga_data/results')
print('End copy of files...')
