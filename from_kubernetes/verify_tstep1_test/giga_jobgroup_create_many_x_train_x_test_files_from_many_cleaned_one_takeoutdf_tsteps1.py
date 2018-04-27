import pandas as pd
import numpy as np

# In[2]:

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


thresholdd = 1

# <hr style="border: 1px solid #f00" />

# <h2 style="color:#2467C0"> 
# 
# [@] Break cleaned files into multiple X_train, Y_train small size files 
# 
# </h2>

# In[28]:

# Label Column ('Status_Failed' is the label column)
label_col = ['Status_Failed']

# In[29]:

thresholdd=1 #Check it matched the title


# In[30]:

# Takes a df and converts it to 3D tensor
# Each sample will have k time steps

def samples_features(df_input):
    
    k = thresholdd
    input_cols = train_feat
    
    # takes a df
    # Put your inputs into a single list
    
    df = pd.DataFrame()
    
    df['single_input_vector'] = df_input[input_cols].apply(tuple, axis=1).apply(list)
    
    # Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
    df['single_input_vector'] = df.single_input_vector.apply(lambda x: [list(x)])
        
    # The starting point
    df['cumulative_input_vectors'] = df['single_input_vector'].shift(0)
    
    for i in range(1,k):
        df['cumulative_input_vectors'] += df['single_input_vector'].shift(i)
          
    df.dropna(inplace=True)     # does operation in place & returns None

    # Extract your training data
    X_ = np.asarray(df.cumulative_input_vectors)
    
    # Use hstack to and reshape to make the inputs a 3d vector
    X = np.vstack(X_).reshape(len(df), k, len(input_cols))
    
    # Clean up
    del df
    
    return X
    # returns 3D array


# In[31]:

# Takes a df and converts it to 3D tensor
# Each sample will have k time steps

def samples_label(df_input):
    
    k = thresholdd
    input_cols = label_col
    
    # takes a df
    # Put your inputs into a single list
    
    df = pd.DataFrame()
    
    df['single_input_vector'] = df_input[input_cols].apply(tuple, axis=1).apply(list)
    
    # Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
    df['single_input_vector'] = df.single_input_vector.apply(lambda x: [list(x)])
        
    # The starting point
    df['cumulative_input_vectors'] = df['single_input_vector'].shift(0)
    
    for i in range(1,k):
        df['cumulative_input_vectors'] += df['single_input_vector'].shift(i)
          
    df.dropna(inplace=True)     # does operation in place & returns None

    # Extract your training data
    X_ = np.asarray(df.cumulative_input_vectors)
    
    # Use hstack to and reshape to make the inputs a 3d vector
    X = np.vstack(X_).reshape(len(df), k, len(input_cols))
    
    # Clean up
    del df
    
    return X
    # returns 3D array


# In[32]:

import pickle

with open('giga_training_features_w_jobgroup.pickle', 'rb') as handle:
    train_feat = pickle.load(handle)


# In[33]:

# Unison shuffle
def unison_shuffled_copies(a, b):
    import numpy as np
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# In[34]:

# Get list of each filename from above so we can go through collector in small batches

from os import listdir
from os.path import isfile, join
path = "."

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles = [x for x in onlyfiles if 'giga_cleaned_w_jobgroup_part' in x]
onlyfiles.sort() #these files contain done HBLI (each file contains a list of dataframes)
setofcleanedfiles=onlyfiles


# In[35]:

import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# In[36]:

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return retLst


# In[37]:

import gc
gc.collect()

# In[42]:

breakeachinto = 15 


# In[38]:

def genereate_training_set():
    counter = 1
    minicounter = 1
    
    # read each cleaned file
    for f in setofcleanedfiles:
        print("File : ", minicounter)
        onedf = pd.read_pickle(f)
        length = onedf.shape[0]
        
        for bat in batch([x for x in range(length)], int(length/breakeachinto)):
            print("Batch : ", counter)
            cleaned_sub    = onedf.iloc[bat]
            cleanedgrouped = cleaned_sub.groupby('JobID')
            print('Collectrain started...')
            #
            collecttrain   = applyParallel(cleanedgrouped, samples_features)
            del cleaned_sub
            X = []
            for x in collecttrain:
                #len(x)
                for i in x:
                    X.append(i)
            del collecttrain
            X = np.array(X)
            #
            print('Collectlabel started...')
            collectlabel = applyParallel(cleanedgrouped, samples_label)
            del cleanedgrouped
            
            Y=[]
            for x in collectlabel:
                #len(x)
                for i in x:
                    Y.append(i)
            del collectlabel
            Y  = np.array(Y)
            YY = np.array([x[0][0] for x in Y]).reshape(len(Y),1)
            # 
            X_final, Y_final = unison_shuffled_copies(X,YY)
            
            gc.collect()
            
            # picle X_final, Y_final
            print('Writing X_final to disk')
            with open('X_final_giga_long_'+'_file_'+str(minicounter)+'_'+str(counter)+'.pickle', 'wb') as handle:
                pickle.dump(X_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            del X_final
            gc.collect()
            
            print('Writing Y_final to disk')
            with open('Y_final_giga_long_'+'_file_'+str(minicounter)+'_'+str(counter)+'.pickle', 'wb') as handle:
                pickle.dump(Y_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            del Y_final
            gc.collect()
            counter +=1
        # end of inner for loop
        del onedf
        #outer for that goes over each file
        minicounter +=1


# In[ ]:

# genereate_training_set()


# <hr style="border: 1px solid #f00" />

# <h2 style="color:#2467C0"> 
# 
# [@] Break takeoutdf into multiple X_test, Y_test small size files
# 
# </h2>

# In[39]:

# Get list of each filename from above so we can go through collector in small batches

from os import listdir
from os.path import isfile, join
path = "."

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles = [x for x in onlyfiles if 'giga_takeoutdf_w_jobgroup' in x]
onlyfiles.sort() #these files contain done HBLI (each file contains a list of dataframes)
setoftakeoutdffiles=onlyfiles

# In[40]:
def genereate_test_set():
    counter = 1
    minicounter = 1
    
    # read each cleaned file
    for f in setoftakeoutdffiles:
        print("File : ", minicounter)
        onedf = pd.read_pickle(f)
        length = onedf.shape[0]
        
        # variable name *cleaned has not been changed here:
        for bat in batch([x for x in range(length)], int(length/breakeachinto)):
            print("Batch : ", counter, " of ", breakeachinto)
            cleaned_sub    = onedf.iloc[bat]
            cleanedgrouped = cleaned_sub.groupby('JobID')
            print('Collectrain started...')
            collecttrain   = applyParallel(cleanedgrouped, samples_features)
            del cleaned_sub
            X = []
            for x in collecttrain:
                #len(x)
                for i in x:
                    X.append(i)
            del collecttrain
            X = np.array(X)
            #
            # pickle X
            print('Collectlabel started...')
            collectlabel = applyParallel(cleanedgrouped, samples_label)
            del cleanedgrouped
            
            Y=[]
            for x in collectlabel:
                #len(x)
                for i in x:
                    Y.append(i)
            del collectlabel
            Y  = np.array(Y)
            YY = np.array([x[0][0] for x in Y]).reshape(len(Y),1)
            # 
            X_test, Y_test = X,YY # no shuffling in test set
            
            gc.collect()
            
            # picle X_final, Y_final
            print('Writing X_test to disk')
            with open('X_test_giga_long_'+'_file_'+str(minicounter)+'_'+str(counter)+'.pickle', 'wb') as handle:
                pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            del X_test
            gc.collect()
            
            print('Writing Y_test to disk')
            with open('Y_test_giga_long_'+'_file_'+str(minicounter)+'_'+str(counter)+'.pickle', 'wb') as handle:
                pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            del Y_test
            
            gc.collect()
            counter +=1
        # end of inner for loop
        del onedf
        #outer for that goes over each file
        minicounter +=1


# In[41]:

genereate_test_set()

# In[ ]:

############ End of breaking cleaned files into smalled X_train, Y_train files
############ End of breaking takeoutdf files into smalled X_test, Y_test files
