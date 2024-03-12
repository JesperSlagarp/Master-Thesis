import os
import sys

path_common=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Common/";
sys.path.append(path_common) #Adds common path to import the python files

from ModelTrainer import ModelTrainer
import tensorflow as tf
from ModelFuncs import get_quipu_model
import math


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')


n_runs=2;
n1=2048;n2=1024;
brow_aug=0.9;
use_brow_aug=True;

lr=5e-4;batch_size=256; #Should keep them constant for all the runs to make a fair comparison

folder_es_train="../results/TrainingWithES/";
mt=ModelTrainer(brow_std=brow_aug,batch_size=batch_size,brow_aug_use=use_brow_aug,lr=lr,opt_aug=False);
model=get_quipu_model(n_dense_1=n1,n_dense_2=n2);
if use_brow_aug:
    run_name="WBrowAug_"+str(int(brow_aug))+str(int(math.modf(brow_aug)[0]*100))+"_N1_"+str(n1)+"_N2_"+str(n2)+"_again.csv";
else:
    run_name="Reproduction"+"_N1_"+str(n1)+"_N2_"+str(n2)+"_again.csv";
mt.crossval_es(model,n_runs=n_runs,data_folder=folder_es_train+run_name)
