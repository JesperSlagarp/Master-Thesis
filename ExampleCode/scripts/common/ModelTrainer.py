# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:04:26 2023

@author: JK-WORK
"""

from DataLoader import DataLoader;
from DataAugmentator import DataAugmentator;
from sklearn.utils import class_weight
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
#from ModelFuncs import get_quipu_model
import time
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import clone_model
from DatasetFuncs import dataset_split
import ipdb


class ModelTrainer():
    def __init__(self, use_brow_aug = False, n_epochs_max=100,lr = 1e-3,batch_size=128,early_stopping_patience=100,use_weights=False,track_losses=False, optimizer="Adam",momentum=None): #Opt_aug still has bugs, have to check
        self.use_brow_aug = use_brow_aug;
        self.dl=DataLoader();
        self.da=DataAugmentator();
        self.shapeX = (-1, QUIPU_LEN_CUT,1); self.shapeY = (-1, QUIPU_N_LABELS);
        self.n_epochs_max=n_epochs_max;
        self.lr=lr;
        self.batch_size=batch_size;
        self.early_stopping_patience=early_stopping_patience;
        self.use_weights=use_weights;
        self.train_losses=[];
        self.valid_losses=[];
        self.train_aug_losses=[];
        self.track_losses=track_losses;
        self.optimizer=optimizer;
        self.momentum=momentum;
    
    def num_list_to_str(self,num_list):
        return '[{:s}]'.format(' '.join(['{:.3f}'.format(x) for x in num_list]))
    
    # Stratified KFold cross validation
    def hpo_crossval_es(self, trial, hypermodel, n_splits=5, data_folder='../../results/QuipuTrainedWithES.csv', save_each_fold=False):
        cols = ["Fold", "Train Acc", "Validation Acc", "Test Acc", "N Epochs", "Runtime"]
        if self.track_losses:
            cols.extend(["Train Losses", "Train Aug Losses", "Valid Losses"])
        df_results = pd.DataFrame(columns=cols)
        
        trainSet, testSet = dataset_split(self.dl.df_cut)
        X_test, Y_test = self.dl.quipu_df_to_numpy(testSet)

        for fold_index in range(n_splits):
            print(f"Starting fold {fold_index + 1}")
            
            X_train, X_valid, Y_train, Y_valid = self.dl.get_datasets_numpy_kfold(trainSet, fold_index, n_splits)
            
            model = hypermodel.build(trial.hyperparameters)

            start_time = time.time()
            train_acc, valid_acc, test_acc, n_epoch = self.crossval_train_es(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            runtime = time.time() - start_time
            
            row = [fold_index + 1, train_acc, valid_acc, test_acc, n_epoch, runtime]
            if self.track_losses:
                row.extend([self.num_list_to_str(self.train_losses), self.num_list_to_str(self.train_aug_losses), self.num_list_to_str(self.valid_losses)])
            temp_df = pd.DataFrame([row], columns=cols)
            df_results.loc[len(df_results)] = row
            
            if save_each_fold:
                row_filename = f"{data_folder[:-4]}_fold{fold_index+1}.csv"
                df_row = pd.DataFrame([row], columns=cols)
                df_row.to_csv(row_filename, index=False)

        df_results.loc[len(df_results)] = df_results.mean(axis=0)

        df_results.to_csv(data_folder, index=False)

        mean_results = df_results.mean(axis=0)
        
        return mean_results['Train Acc'], mean_results['Validation Acc'], mean_results['Test Acc'], mean_results['N Epochs']

    def crossval_train_es(self, model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batch_size_val=512):

        # Reshape the validation datasets to fit the model's input shape
        X_valid_rs = X_valid.reshape(self.shapeX)
        Y_valid_rs = Y_valid.reshape(self.shapeY)

        # Initialize variables for early stopping and best model weights tracking
        best_weights = model.get_weights()
        best_valid_loss = 1e6
        patience_count = 0

        # Compute class weights for balancing if necessary
        weights = class_weight.compute_class_weight('balanced', classes=np.arange(QUIPU_N_LABELS), y=np.argmax(Y_train, axis=1))
        weights = dict(zip(np.arange(QUIPU_N_LABELS), weights))
        weights_final = weights if self.use_weights else None

        # Initialize lists for tracking losses
        self.train_losses = []
        self.valid_losses = []
        self.train_aug_losses = []

        # Training loop with early stopping
        for n_epoch in range(self.n_epochs_max):
            print("=== Epoch:", n_epoch, "===")
            start_time = time.time()

            # Apply data augmentation if specified
            X = self.da.all_augments(X_train) if self.use_brow_aug else self.da.quipu_augment(X_train)
            preparation_time = time.time() - start_time

            # Fit the model for one epoch
            out_history = model.fit(x=X.reshape(self.shapeX), y=Y_train.reshape(self.shapeY), batch_size=self.batch_size,
                                    shuffle=True, epochs=1, verbose=1, class_weight=weights_final)

            # Evaluate on the validation dataset
            print("Validation ds:")
            valid_res = model.evaluate(x=X_valid_rs, y=Y_valid_rs, verbose=True, batch_size=batch_size_val)
            if valid_res[0] < best_valid_loss:
                best_valid_loss = valid_res[0]
                patience_count = 0
                best_weights = model.get_weights()
            else:
                patience_count += 1

            # Track preparation and training time
            training_time = time.time() - start_time - preparation_time
            print('  prep time: %3.1f sec' % preparation_time, '  train time: %3.1f sec' % training_time)

            # Early stopping condition
            if patience_count > self.early_stopping_patience or n_epoch == self.n_epochs_max - 1:
                print("Stopping learning because of early stopping:")
                model.set_weights(best_weights)
                break

            # Optionally track losses for analysis
            if self.track_losses:
                self.valid_losses.append(valid_res[0])
                train_res = model.evaluate(x=X_train.reshape(self.shapeX), y=Y_train, batch_size=batch_size_val)
                self.train_losses.append(train_res[0])
                self.train_aug_losses.append(out_history.history['loss'][0])

        # Evaluate the model on training, validation, and test datasets to get the final accuracies
        train_acc, valid_acc, test_acc = self.eval_model_and_print_results(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
        return train_acc, valid_acc, test_acc, n_epoch

    def hpo_train_es(self, hp, model, batch_size_val=512): #Runs training with early stopping, more controlled manner than quipus original

        #model.compile(
        #        optimizer=Adam(learning_rate=hp.get('lr')),
        #        loss="categorical_crossentropy",
        #        metrics=["accuracy"],
        #    )

        #model.build(hp)

        X_train,X_valid,Y_train,Y_valid,X_test,Y_test=self.dl.get_datasets_numpy(repeat_classes= (not self.use_weights) ); #When weights are used 
        
        X_valid_rs = X_valid.reshape(self.shapeX); Y_valid_rs = Y_valid.reshape(self.shapeY)
        best_weights=model.get_weights();best_valid_loss=1e6;patience_count=0;
        weights=class_weight.compute_class_weight(class_weight ='balanced',classes = np.arange(QUIPU_N_LABELS), y =np.argmax(Y_train,axis=1))
        weights=dict(zip(np.arange(QUIPU_N_LABELS), weights))
        weights_final= weights if self.use_weights else None;
        self.train_losses=[];
        self.valid_losses=[];
        self.train_aug_losses=[];
        #ipdb.set_trace();
        for n_epoch in range(self.n_epochs_max):
            print("=== Epoch:", n_epoch,"===")
            start_time = time.time()
            X= self.da.all_augments(X_train) if self.use_brow_aug else self.da.quipu_augment(X_train);
            preparation_time = time.time() - start_time
            # Fit the model
            out_history = model.fit( 
                x = X.reshape(self.shapeX), y = Y_train.reshape(self.shapeY), 
                batch_size=self.batch_size, shuffle = True, epochs=1,verbose = 1, class_weight = weights_final, 
            )
            
            print("Validation ds:")
            valid_res=model.evaluate(x = X_valid_rs,   y = Y_valid_rs,   verbose=True,batch_size=batch_size_val);
            if valid_res[0]<best_valid_loss:
                best_valid_loss=valid_res[0]
                patience_count=0;
                best_weights=model.get_weights()
            else:
                patience_count+=1;
            #Others
            training_time = time.time() - start_time - preparation_time
            if self.track_losses:
                self.valid_losses.append(valid_res[0]);
                train_res=model.evaluate(x = X_train.reshape(self.shapeX),   y = Y_train, batch_size=batch_size_val);
                self.train_losses.append(train_res[0]);
                self.train_aug_losses.append(out_history.history['loss'][0]);
            
            print('  prep time: %3.1f sec' % preparation_time, '  train time: %3.1f sec' % training_time)
            if patience_count>self.early_stopping_patience or n_epoch==self.n_epochs_max-1:
                print("Stopping learning because of early stopping:")
                model.set_weights(best_weights)
                break
        train_acc,valid_acc,test_acc=self.eval_model_and_print_results(model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
        return train_acc, valid_acc, test_acc, n_epoch
    
    def train_es(self, model, batch_size_val=512): #Runs training with early stopping, more controlled manner than quipus original

        X_train,X_valid,Y_train,Y_valid,X_test,Y_test=self.dl.get_datasets_numpy(repeat_classes= (not self.use_weights) ); #When weights are used 
        if self.optimizer=="Adam":
            model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=self.lr),metrics = ['accuracy'])
        else:
            model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate=self.lr,momentum=self.momentum),metrics = ['accuracy'])
        X_valid_rs = X_valid.reshape(self.shapeX); Y_valid_rs = Y_valid.reshape(self.shapeY)
        best_weights=model.get_weights();best_valid_loss=1e6;patience_count=0;
        weights=class_weight.compute_class_weight(class_weight ='balanced',classes = np.arange(QUIPU_N_LABELS), y =np.argmax(Y_train,axis=1))
        weights=dict(zip(np.arange(QUIPU_N_LABELS), weights))
        weights_final= weights if self.use_weights else None;
        self.train_losses=[];
        self.valid_losses=[];
        self.train_aug_losses=[];
        #ipdb.set_trace();
        for n_epoch in range(self.n_epochs_max):
            print("=== Epoch:", n_epoch,"===")
            start_time = time.time()
            X= self.da.all_augments(X_train) if self.use_brow_aug else self.da.quipu_augment(X_train);
            preparation_time = time.time() - start_time
            # Fit the model
            out_history = model.fit( 
                x = X.reshape(self.shapeX), y = Y_train.reshape(self.shapeY), 
                batch_size=self.batch_size, shuffle = True, epochs=1,verbose = 1, class_weight = weights_final, 
            )
            
            print("Validation ds:")
            valid_res=model.evaluate(x = X_valid_rs,   y = Y_valid_rs,   verbose=True,batch_size=batch_size_val);
            if valid_res[0]<best_valid_loss:
                best_valid_loss=valid_res[0]
                patience_count=0;
                best_weights=model.get_weights()
            else:
                patience_count+=1;
            #Others
            training_time = time.time() - start_time - preparation_time
            if self.track_losses:
                self.valid_losses.append(valid_res[0]);
                train_res=model.evaluate(x = X_train.reshape(self.shapeX),   y = Y_train, batch_size=batch_size_val);
                self.train_losses.append(train_res[0]);
                self.train_aug_losses.append(out_history.history['loss'][0]);
            
            print('  prep time: %3.1f sec' % preparation_time, '  train time: %3.1f sec' % training_time)
            if patience_count>self.early_stopping_patience or n_epoch==self.n_epochs_max-1:
                print("Stopping learning because of early stopping:")
                model.set_weights(best_weights)
                break
        train_acc,valid_acc,test_acc=self.eval_model_and_print_results(model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
        return train_acc, valid_acc, test_acc, n_epoch
    
    ##Quipu base code to compare
    def quipu_def_train(self,model,n_epochs=60,with_brow_aug=False):
        #tensorboard, history = resetHistory()
        lr = 1e-3
        X_train,X_valid,Y_train,Y_valid,X_test,Y_test=self.dl.get_datasets_numpy_quipu();
        model.compile(
            loss = 'categorical_crossentropy', 
            optimizer = Adam(lr=0.001),
            metrics = ['accuracy']
        )

        weights=class_weight.compute_class_weight(class_weight ='balanced',classes = np.arange(QUIPU_N_LABELS), y =np.argmax(Y_train,axis=1))
        weights=dict(zip(np.arange(QUIPU_N_LABELS), weights))
        for n in range(0, n_epochs):
            print("=== Epoch:", n,"===")
            start_time = time.time()
           
            X=self.da.all_augments(X_train) if with_brow_aug else self.da.quipu_augment(X_train) ;
            # Learning rate decay
            lr = lr*0.97
            model.optimizer.lr.assign(lr)
            preparation_time = time.time() - start_time
            # Fit the model
            out_history = model.fit( 
                x = X.reshape(self.shapeX), 
                y = Y_train.reshape(self.shapeY), 
                batch_size=32, shuffle = True,
                initial_epoch = n,  epochs=n+1,
                class_weight = weights, 
                validation_data=(X_valid.reshape(self.shapeX),  Y_valid.reshape(self.shapeY)),
                #callbacks = [tensorboard, history], 
                verbose = 0
            )
            training_time = time.time() - start_time - preparation_time
            
            # Feedback 
            print('  prep time: %3.1f sec' % preparation_time, 
                  '  train time: %3.1f sec' % training_time)
            print('  loss: %5.3f' % out_history.history['loss'][0] ,'  acc: %5.4f' % out_history.history['accuracy'][0] ,'  val_acc: %5.4f' % out_history.history['val_accuracy'][0])
            #print('  loss: %5.3f' % out_history.history['loss'][0] ,'  acc: %5.4f' % out_history.history['accuracy'][0] ,'  val_acc: %5.4f' % out_history.history['val_acc'][0])
        train_acc,valid_acc,test_acc=self.eval_model_and_print_results(model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
        return train_acc, valid_acc, test_acc
    
    def eval_model_and_print_results(self,model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
        print("       [ loss , accuracy ]")
        train_results= model.evaluate(x = X_train.reshape(self.shapeX), y = Y_train, verbose=False);
        valid_results=model.evaluate(x = X_valid.reshape(self.shapeX),   y = Y_valid,   verbose=False)
        test_results= model.evaluate(x = X_test.reshape(self.shapeX),  y = Y_test,  verbose=False)
        
        print("Train:", train_results )
        print("Validation  :", valid_results )
        print("Test :", test_results )
        train_acc= train_results[1];valid_acc= valid_results[1];test_acc= test_results[1];
        return train_acc,valid_acc,test_acc

    
#if __name__ == "__main__":
#    import tensorflow as tf
#    physical_devices = tf.config.list_physical_devices('GPU')
#    tf.config.set_visible_devices(physical_devices[0], 'GPU')
#    mt=ModelTrainer();
#    model=get_quipu_model();
#    mt.n_epochs_max=3;
#    mt.crossval_es(model,n_runs=2)
