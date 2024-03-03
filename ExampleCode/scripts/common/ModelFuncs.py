# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:23:22 2023

@author: JK-WORK
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D, GlobalAveragePooling1D, Dense, ReLU
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
from tensorflow.keras.regularizers import l2
import ModelTrainer
import keras_tuner

class fcnHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
        
        filters_block_0 = hp.Int(f"filters_block_0", min_value=64, max_value=256, step=64)
        kernel_size_block_0 = hp.Int(f"kernel_size_block_0", min_value=3, max_value=7, step=2)
        x = Conv1D(filters_block_0, kernel_size_block_0, padding='same')(input_trace)
        x = layers.BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

        for i in range(hp.Int("num_layers", 2, 5)):
            filters_block = hp.Int(f"filters_block_{i + 1}", min_value=64, max_value=256, step=64)
            kernel_size_block = hp.Int(f"kernel_size_block_{i + 1}", min_value=3, max_value=7, step=2)
            x = Conv1D(filters_block, kernel_size_block, padding='same')(x)
            x = layers.BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        
        x = GlobalAveragePooling1D()(x)

        output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
        model = Model(inputs=input_trace, outputs=output_barcode)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        #model.compile(
        #    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        #    loss="categorical_crossentropy",
        #    metrics=["accuracy"],
        #)
        return model
    
    def fit(self, hp, model, training_function, **kwargs):
        train_acc, valid_acc, test_acc, n_epoch = training_function(hp, model)
        return (-test_acc)


def create_fcn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, 8, padding='same', input_shape=input_shape),
        BatchNormalization(),
        ReLU(),
        Conv1D(256, 5, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv1D(128, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv1D(128, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        GlobalAveragePooling1D(),
        Dense(num_classes, activation='softmax')
    ])
    return model

##### QUIPUNET ###############
def get_quipu_model(n_filters_block_1=64,kernel_size_block_1=7,dropout_intermediate_blocks=0.25,
                    n_filters_block_2=128,kernel_size_block_2=5,n_filters_block_3=256,kernel_size_block_3=3,
                    n_dense_1=512,n_dense_2=512,dropout_final=0.4):
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')

    x = Conv1D(n_filters_block_1, kernel_size_block_1, padding="same")(input_trace)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_1, kernel_size_block_1, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x) 
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Conv1D(n_filters_block_2, kernel_size_block_2, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_2, kernel_size_block_2, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Conv1D(n_filters_block_3, kernel_size_block_3, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv1D(n_filters_block_3, kernel_size_block_3, padding="same")(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(dropout_intermediate_blocks)(x)
    
    x = Flatten()(x)
    x = Dense(n_dense_1, activation='relu')(x)
    x = Dropout(dropout_final)(x)
    x = Dense(n_dense_2, activation='relu')(x)
    x = Dropout(dropout_final)(x)
    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model;
'''
Total params: 15,671,512
Trainable params: 15,667,472
Non-trainable params: 4,040  When Ndense1=2048 Ndense2=1024. To know how many params to start resnets with. 
'''
def get_quipu_skipCon_model(filter_size=64,kernels_blocks=[7,5,3],dropout_blocks=0.25,n_dense_1=512,n_dense_2=512,dropout_final=0.4,pool_size=3,activation="relu"):
    modelInfo=ModelInfo(model_type="QuipuSkip",filter_size=filter_size,kernels_blocks=kernels_blocks,dense_1=n_dense_1,dense_2=n_dense_2,dropout_end=dropout_final,dropout_blocks=dropout_blocks,activation=activation);
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
    x=input_trace;
    for i in range(len(kernels_blocks)):
        x = quipu_block_skip_con(x,filter_size,kernels_blocks[i],pool_size,dropout_blocks,activation)
        filter_size*=2;
    x = Flatten()(x)
    x = Dense(n_dense_1, activation=activation)(x)
    x = Dropout(dropout_final)(x)
    x = Dense(n_dense_2, activation=activation)(x)
    x = Dropout(dropout_final)(x)
    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model,modelInfo;

def quipu_block_skip_con(x,n_filters,kernel_size,pool_size,dropout_val,activation):
    x_skip = Conv1D(n_filters, 1, padding = 'same')(x)
    conv_path = Conv1D(n_filters, kernel_size, padding="same")(x)
    conv_path = layers.BatchNormalization(axis=1)(conv_path)
    conv_path = Activation(activation)(conv_path)
    conv_path = Conv1D(n_filters, kernel_size, padding="same")(conv_path)
    conv_path = layers.BatchNormalization(axis=1)(conv_path)
    out = layers.add([conv_path, x_skip])  
    out = Activation(activation)(out)
    out = MaxPooling1D(pool_size=pool_size)(out)
    out = Dropout(dropout_val)(out)
    return out
    
##### ResNetBased ###############
class  ModelInfo:
    def __init__(self,model_type="ResNet",filter_size=2,block_layers=[1,1],dense_1=None,dense_2=None,dropout_end=0,dropout_blocks=0,kernels_blocks=0,activation="relu"):
        self.model_type=model_type;
        self.filter_size=filter_size;
        self.block_layers=block_layers;
        self.dense_1=dense_1
        self.dense_2=dense_2
        self.dropout_end=dropout_end;
        self.dropout_blocks=dropout_blocks;
        self.kernels_blocks=kernels_blocks;
        self.activation=activation;
#based on https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
def get_resnet_model(filter_size=64, block_layers=[3,4,6,3], init_conv_kernel=7,init_pool_size=3, end_pool_size=2,dense_1=None,dropout_end=0.3,l2reg=None,dense_2=None,activation_fnc='relu',dropout_block=None):
    modelInfo=ModelInfo(model_type="ResNet",filter_size=filter_size,block_layers=block_layers,dense_1=dense_1,dense_2=dense_2,dropout_end=dropout_end,dropout_blocks=dropout_block,activation=activation_fnc);
    kernel_regularizer=None if (l2reg is None) else l2(l2reg);
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
    
    x = Conv1D(filter_size, init_conv_kernel, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(input_trace)
    x = BatchNormalization()(x)
    x = Activation(activation_fnc)(x)
    x = MaxPooling1D(pool_size=init_pool_size,strides=2, padding = 'same')(x)
    
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = resnet_identity_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc,dropout=dropout_block)
        else:
            # One Residual/Convolutional B
            filter_size = filter_size*2# The filter size will go on increasing by a factor of 2
            x = resnet_conv_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc,dropout=dropout_block)
            for j in range(block_layers[i] - 1):
                x = resnet_identity_block(x, filter_size,kernel_regularizer=kernel_regularizer,activation_str=activation_fnc,dropout=dropout_block)
    
    x = AveragePooling1D(pool_size=end_pool_size, padding = 'same')(x)
    x=Flatten()(x)
    
    if dense_1 is not None:
        x=Dense(dense_1, activation=activation_fnc,kernel_regularizer=kernel_regularizer)(x)
        x=Dropout(dropout_end)(x)
    if dense_2 is not None:
        x=Dense(dense_2, activation=activation_fnc,kernel_regularizer=kernel_regularizer)(x)
        x=Dropout(dropout_end)(x)
    if (dense_1 is None) and (dense_2 is None):
        x=Dropout(dropout_end)(x)

    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
    model = Model(inputs=input_trace, outputs=output_barcode)
    return model,modelInfo;
def resnet_identity_block(x,filter_size,kernel_size=3,kernel_regularizer=None,activation_str='relu',dropout=None):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',kernel_regularizer=kernel_regularizer)(x)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Activation(activation_str)(conv_block)
    # Layer 2
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',kernel_regularizer=kernel_regularizer)(conv_block)
    conv_block = BatchNormalization()(conv_block)
    if not (dropout is None):
        conv_block=Dropout(dropout)(conv_block)
    # Add Residue
    out = layers.add([conv_block, x_skip])     
    out = Activation(activation_str)(out)
    return out

def resnet_conv_block(x,filter_size,kernel_size=3,kernel_regularizer=None,activation_str='relu',dropout=None):
    # Layer 1
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(x)
    conv_block = BatchNormalization()(conv_block)
    conv_block = Activation(activation_str)(conv_block)
    # Layer 2
    conv_block = Conv1D(filter_size, kernel_size, padding = 'same')(conv_block)
    conv_block = BatchNormalization()(conv_block)
    if not (dropout is None):
        conv_block=Dropout(dropout)(conv_block)
    # Add Residue
    x_skip = Conv1D(filter_size, 1, padding = 'same',strides=2,kernel_regularizer=kernel_regularizer)(x) ##Kernel for skip connection is 1
    out = layers.add([conv_block, x_skip])     
    out = Activation(activation_str)(out)
    return out

##### QUIPUNET WITH SKIP CONNECTIONS ###########

if __name__ == "__main__":
    model,modelInfo=get_quipu_skipCon_model(filter_size=64,kernels_blocks=[7,5,3],dropout_blocks=0.25,n_dense_1=512,n_dense_2=512,dropout_final=0.4,pool_size=3,activation="relu");
    model.summary();

