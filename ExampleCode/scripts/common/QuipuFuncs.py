import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from scipy import signal


##Augmentations
def addNoise(xs, std = 0.05):
    return xs + np.random.normal(0, std, xs.shape)

def stretchDuration(xs, std = 0.1, probability = 0.5):
    """
    Augment the length by re-sampling. probability gives ratio of mutations
    Slow method since it uses scipy
    
    :param xs: input numpy data structure
    :param std: amound to mutate (drawn from normal distribution)
    :param probability: probabi
    """
    x_new = np.copy(xs)
    for i in range(len(xs)):
        if np.random.rand() > probability:
            x_new[i] = _mutateDurationTrace(x_new[i], std)
    return x_new

def _mutateDurationTrace(x, std = 0.1):
    "adjust the sampling rate"
    length = len(x)
    return normaliseLength( signal.resample(x, int(length*np.random.normal(1, std))) , length = length)
    
def magnitude(xs, std = 0.15):
    "Baseline mutation"
    return xs * np.abs(np.random.normal(1, std, len(xs)).reshape((-1,1)) ) 

###

### From tools
def noiseLevels(train = None):
    """
    Gives typical noise levels in the system 
    
    :param train: data to train on (numpy array)
    :return: typical noise levels (default: 0.006)
    """
    global constantTypicalNoiseLevels
    #constantTypicalNoiseLevels = 4
    if ~('constantTypicalNoiseLevels' in dir()):
        constantTypicalNoiseLevels = 0.006 # default
    if train is not None:
        tmp = np.array( list(map(np.std, train)) );
        constantTypicalNoiseLevels = tmp[~np.isnan(tmp)].mean()
    return constantTypicalNoiseLevels

def normaliseLength(trace, length = 600, trim = 0):
    """
    Normalizes the length of the trace and trims the front 
    
    :param length: length to fit the trace into (default: 600)
    :param trim: how many points to drop in front of the trace (default: 0)
    :return: trace of length 'length' 
    """
    if len(trace) >= length + trim:
        return trace[trim : length+trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, noiseLevels(), length - len(trace[trim:]))
        )    
########

def barcodeToNumber(barcode):
    "translates the barcode string into number"
    if len(np.shape(barcode)) == 0 :
        return barcodeEncoding[barcode]
    elif len(np.shape(barcode)) == 1:
        fn = np.vectorize(lambda key: barcodeEncoding[key])
        return fn(barcode)
    elif len(np.shape(barcode)) == 2 and np.shape(barcode)[1] == 1:
        return barcodeToNumber(np.reshape(barcode, (-1,)))
    else:
        raise ValueError("Error: wrong input recieved: "+str(barcode))

def numberToBarcode(number):
    "number to barcode string"
    if len(np.shape(number)) == 0 :
        return barcodeEncodingReverse[number]
    elif len(np.shape(number)) == 1:
        fn = np.vectorize(lambda key: barcodeEncodingReverse[key])
        return fn(number)
    else:
        raise ValueError("Error: wrong input recieved: "+str(number))
    
def numberToOneHot(number):
    return keras.utils.to_categorical(number, num_classes= hp["barcodes"])
    
def oneHotToNumber(onehot):
    if np.shape(onehot) == (hp['barcodes'],):
        return np.argmax(onehot)
    elif len(np.shape(onehot)) == 2 and np.shape(onehot)[1] == hp['barcodes']:
        return np.apply_along_axis(arr=onehot, func1d=np.argmax, axis=1)
    else:
        raise ValueError("Error: wrong input recieved: "+str(onehot))

    
def barcodeToOneHot(barcode):
    "barcode string to catogory encoding aka One-Hot"
    return numberToOneHot( barcodeToNumber(barcode) )
    
def oneHotToBarcode(onehot):
    "catogory encoding aka One-Hot to barcode string"
    return numberToBarcode( oneHotToNumber(onehot) )
    
def genNumpyDataset(dataset):
    
#Hardcoded parameters
# Hyperparameters

hp = {
    "traceLength" : 700,
    "traceTrim"   : 0,
    "barcodes"    : 8,        # distinct barcode count 
    "normalise_levels": True, # wherther to normalise experiments per batch before feetingh into NN
}


# # barcode binnary encoding: int(barcode,2) 
# # explicit for customistation 
barcodeEncoding = {
    "000" : 0,
    "001" : 1,
    "010" : 2,
    "011" : 3,
    "100" : 4,
    "101" : 5,
    "110" : 6,
    "111" : 7
}

