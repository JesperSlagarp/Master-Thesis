import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from params import QUIPU_DATA_FOLDER,QUIPU_VALIDATION_PROP_DEF,QUIPU_N_LABELS
from DatasetFuncs import allDataset_loader,dataset_split
import ipdb

class DataLoader():
    def __init__(self,min_perc_test=4,max_perc_test=15,reduce_dataset_samples=None):
        self.min_perc_test=min_perc_test;
        self.max_perc_test=max_perc_test; #Porcentages for the split between test and train ds.
        self.reduce_dataset_samples=reduce_dataset_samples;
        self.df_cut=allDataset_loader(QUIPU_DATA_FOLDER,cut=True); #Loads both datasets to the class
        self.df_uncut=allDataset_loader(QUIPU_DATA_FOLDER,cut=False);
        
    def get_datasets_numpy(self,validation_prop=0.15,repeat_classes=True): #Gets the numpy arrays for the NN repeating samples per class so all have the same "weight", and also separates into validation keeping the same percentage of samples per class.
        df_train,df_test=dataset_split(self.df_cut,min_perc=self.min_perc_test,max_perc=self.max_perc_test);
        X_train,Y_train=self.quipu_df_to_numpy(df_train);X_test,Y_test=self.quipu_df_to_numpy(df_test);
        X_train,X_valid,Y_train,Y_valid=self.divide_numpy_ds(X_train,Y_train,1-validation_prop,keep_perc_classes=True,repeat_classes=repeat_classes);
        return X_train,X_valid,Y_train,Y_valid,X_test,Y_test

    def get_datasets_numpy_kfold(self, dataset, fold_index, n_splits):

        X, Y = self.quipu_df_to_numpy(dataset)
    
        sgkf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y

        fold_counter = 0
        for train_index, test_index in sgkf.split(X, y_labels):
            if fold_counter == fold_index:
                X_train, Y_train = X[train_index], Y[train_index]
                X_valid, Y_valid = X[test_index], Y[test_index]
                return X_train, X_valid, Y_train, Y_valid
            fold_counter += 1

        
    def get_datasets_numpy_quipu(self,validation_prop=QUIPU_VALIDATION_PROP_DEF): #Gets the numpy arrays for the NN as it is done in Quipus code, with train, validation and test sets
        df_train,df_test=dataset_split(self.df_cut,min_perc=self.min_perc_test,max_perc=self.max_perc_test);
        X_train,Y_train=self.quipu_df_to_numpy(df_train);X_test,Y_test=self.quipu_df_to_numpy(df_test);
        X_train,X_valid,Y_train,Y_valid=self.divide_numpy_ds(X_train,Y_train,1-validation_prop);
        return X_train,X_valid,Y_train,Y_valid,X_test,Y_test
        
    def quipu_df_to_numpy(self,df): # dataframe data structure to numpy arrays, and barcodes in onehot encoding.
        X_numpy=np.vstack( df.trace )
        Y_barcode = np.vstack( df.barcode.values )
        Y_label=np.asarray([int(str(i[0]),2) for i in Y_barcode]);
        Y_onehot = np.zeros((Y_label.size, Y_label.max() + 1))
        Y_onehot[np.arange(Y_label.size), Y_label] = 1
        return X_numpy,Y_onehot;
    
    def divide_numpy_ds(self,X,Y,prop,keep_perc_classes=False,repeat_classes=False): #Divides train in train and validation. Prop indicates proportion of train ds
    #keep perc classes assures that the classes are equally percentally distributed in train and valid dataset.
    #repeat_classes repeats reads so each class has the same amount of samples
        ni_x1 = int( len(X)*prop ) # Training set length
        ni_x2   = len(X) - ni_x1  # Validation set length
        if (keep_perc_classes == False):
            indexes_to_partition = np.arange(len(X))
            np.random.shuffle(indexes_to_partition)  # Shuffles indexes of random
            X1=X[indexes_to_partition[:ni_x1],:] #Based on the random indexes picks the datasets
            Y1=Y[indexes_to_partition[:ni_x1],:]
            X2=X[indexes_to_partition[ni_x1:],:]
            Y2=Y[indexes_to_partition[ni_x1:],:]
        else :
            idxs_train=[];idxs_valid=[]
            #ipdb.set_trace(); 
            Y_labels=np.argwhere(Y==1)[:,1] ##Only to check classes
            values, counts = np.unique(Y_labels, return_counts=True)
            for i in range(QUIPU_N_LABELS):
                all_idxs_code=np.argwhere(np.argwhere(Y==1)[:,1]==i)[:,0] #All indexes of that code. Inside argwhere is to get the label from the one hot encoding.
                np.random.shuffle(all_idxs_code)
                n_train_code=int(prop*len(all_idxs_code));
                idxs_train.append(all_idxs_code[:n_train_code]);idxs_valid.append(all_idxs_code[n_train_code:])
            
            if repeat_classes:
                idxs_train=self.repeat_reads_to_balance(idxs_train)
            idxs_train=np.concatenate(idxs_train);idxs_valid=np.concatenate(idxs_valid); #List of lists to numpy array
            X1=X[idxs_train,:];Y1=Y[idxs_train,:];
            X2=X[idxs_valid,:];Y2=Y[idxs_valid,:];
        
        return X1,X2,Y1,Y2
    def repeat_reads_to_balance(self,idxs_train_in):
        n_evs_per_class = np.asarray([len(i) for i in idxs_train_in]);
        n_evs_goal = np.max(n_evs_per_class);
        idxs_train_out=[np.resize(i,(n_evs_goal,)) for i in idxs_train_in]
        return idxs_train_out;
    def test_balanced_datasets(self,Y_train,Y_valid):
        #ipdb.set_trace();
        Y_train_labels=np.argwhere(Y_train==1)[:,1]
        Y_valid_labels=np.argwhere(Y_valid==1)[:,1]
        values, counts = np.unique(Y_train_labels, return_counts=True)
        values, counts = np.unique(Y_valid_labels, return_counts=True)

if __name__ == "__main__":
    dl=DataLoader();
    #X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy_quipu();
    X_train,X_valid,Y_train,Y_valid,X_test,Y_test=dl.get_datasets_numpy();
    dl.test_balanced_datasets(Y_train,Y_valid);
    
