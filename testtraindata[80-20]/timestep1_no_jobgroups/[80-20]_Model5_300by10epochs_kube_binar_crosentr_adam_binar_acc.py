from keras.utils import multi_gpu_model
import os
import pandas as pd
import numpy as np
import sys
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
import pickle 
from sklearn.metrics import precision_recall_fscore_support as score

# Declare number of GPUs to make visible to your script
pickgpus = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]=pickgpus

 # Read X_final and Y_final
with open('X_final8020.pickle', 'rb') as handle:
    X_final = pickle.load(handle)

with open('Y_final8020.pickle', 'rb') as handle:
    Y_final = pickle.load(handle)

 # Read X_test and Y_test
with open('X_test8020.pickle', 'rb') as handle:
    X_test = pickle.load(handle)

with open('Y_test8020.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)

input_length    = X_final.shape[1]
input_dim       = X_final.shape[2]
output_dim      = len(Y_final[0])

# Model 5

def create_model5(input_dim = input_dim, input_length = input_length, output_dim=output_dim):
    print ('Creating model 5...')
    model = Sequential()
    model.add(LSTM(500, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(500, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(500, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(500, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(500, input_shape=(input_length,input_dim),return_sequences=True))
    model.add(LSTM(500, ))
    model.add(Dense(output_dim, activation='sigmoid'))
    ###
    ###
    print('Initiating parallel GPU model')
    parallel_model = multi_gpu_model(model, gpus=1+pickgpus.count(","))
    print ('Compiling...')
    parallel_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    return parallel_model


model       = create_model5()

for i in range(60*5):
	history     = model.fit(X_final,Y_final,batch_size=8*1024*2, epochs=10, validation_split = 0.10, shuffle=True, verbose = 1)
	loss, accuracy = model.evaluate(X_test, Y_test, batch_size=8*1024*2)
	print('Loss and Accuracy: (iteration#,loss,accuray) ',i, loss, accuracy)
	#Save
	model.save_weights('[80-20]_Model5_weights_every_10epochs_kube_binar_crosentr_adam_binar_acc_300times.h5')

print("Start: Predicting on X_test and making a vector")
y_pred      = model.predict(X_test, batch_size = 8*1024*2)
y_true      = pd.Series([x[0] for x in Y_test])
y_predicted = pd.Series([ np.rint(j[0]) for j in y_pred])
print("End: Predicting on X_test and making a vector")

print("Score:")
precision, recall, fscore, support = score(y_true, y_predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("Confusion matrix:")
print(pd.crosstab(y_true, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True))
