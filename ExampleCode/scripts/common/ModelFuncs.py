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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D, GlobalAveragePooling1D, Dense, ReLU, Add, Reshape, LSTM, concatenate, LayerNormalization, MultiHeadAttention
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
from tensorflow.keras.regularizers import l2
import ModelTrainer
import keras_tuner

def build_model(hp):
    dnn_layers_ss = [1,2,3]
    dnn_units_min, dnn_units_max = 32, 512
    dropout_ss = [0.1, 0.2]
    active_func_ss = ['relu', 'tanh']
    optimizer_ss = ['adam']
    lr_min, lr_max = 1e-4, 1e-1
    
    active_func = hp.Choice('activation', active_func_ss)
    optimizer = hp.Choice('optimizer', optimizer_ss)
    lr = hp.Float('learning_rate', min_value=lr_min, max_value=lr_max, sampling='log')
    
    input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
    flatten_layer = tf.keras.layers.Flatten()(input_trace)
    
    # create hidden layers
    dnn_units = hp.Int(f"0_units", min_value=dnn_units_min, max_value=dnn_units_max)
    dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(flatten_layer)
    for layer_i in range(hp.Choice("n_layers", dnn_layers_ss) - 1):
        dnn_units = hp.Int(f"{layer_i}_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(dense)
        if hp.Boolean("dropout"):
            dense = tf.keras.layers.Dropout(rate=0.25)(dense)
    output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(dense)
    model = Model(inputs=input_trace, outputs=output_barcode)
    
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise("Not supported optimizer")
        
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

class fcnHyperModel(keras_tuner.HyperModel):
    def build(self, hp):

        input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')
        x = input_trace

        min_layers = 3
        max_layers = 6
        num_layers = hp.Int("num_layers", min_layers, max_layers)

        for i in range(max_layers): #Because keras_tuner seems to have a bug
            hp.Choice(f"filters_block_{i}", [32, 64, 128, 256])
            hp.Int(f"kernel_size_block_{i}", min_value=3, max_value=7, step=2)
        
        for i in range(num_layers):
            x = Conv1D(hp.get(f"filters_block_{i}"), hp.get(f"kernel_size_block_{i}"), padding='same')(x)
            x = layers.BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        
        x = GlobalAveragePooling1D()(x)

        output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
        model = Model(inputs=input_trace, outputs=output_barcode)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    
    def fit(self, hp, model, training_function, **kwargs):
        model.summary()
        train_acc, valid_acc, test_acc, n_epoch = training_function(hp, model)
        return {'Accuracy': test_acc}
    
class resNetHyperModel(keras_tuner.HyperModel):

    def build(self, hp):

        input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')

        x = input_trace

        num_blocks = hp.Int("num_blocks", min_value = 2, max_value = 4, step = 1)

        for i in range(num_blocks):
            filters = hp.Int(f"filters_block_{i}", min_value = 32, max_value = 256, step = 32)
            kernel_size_0 = hp.Int(f"kernel_size_0_block_{i}", min_value = 3, max_value = 11, step = 2)
            kernel_size_1 = hp.Int(f"kernel_size_1_block_{i}", min_value = 3, max_value = 11, step = 2)
            kernel_size_2 = hp.Int(f"kernel_size_2_block_{i}", min_value = 3, max_value = 11, step = 2)
            x = self.residual_block(x, filters=filters, kernel_size_0=kernel_size_0, kernel_size_1 = kernel_size_1, kernel_size_2 = kernel_size_2)

        x = GlobalAveragePooling1D()(x)
        output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
        model = Model(inputs=input_trace, outputs=output_barcode)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        return model
    
    @staticmethod
    def residual_block(input_tensor, filters, kernel_size_0, kernel_size_1, kernel_size_2):
        # First component of the main path
        x = Conv1D(filters=filters, kernel_size=kernel_size_0, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second component
        x = Conv1D(filters=filters, kernel_size=kernel_size_1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Third component
        x = Conv1D(filters=filters, kernel_size=kernel_size_2, padding='same')(x)
        x = BatchNormalization()(x)

        # Shortcut path
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        # Adding shortcut to the main path
        x = Add()([x, shortcut])
        x = ReLU()(x)
        
        return x
    
    def fit(self, hp, model, training_function, **kwargs):
        train_acc, valid_acc, test_acc, n_epoch = training_function(hp, model)
        return {'Accuracy': test_acc}
    
class fcnLstmHyperModel(keras_tuner.HyperModel):

    def build(self, hp):

        input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')

        # FCN Block
        kernel_size_0 = hp.Int(f"kernel_size_0", min_value = 3, max_value = 11, step = 2)
        kernel_size_1 = hp.Int(f"kernel_size_1", min_value = 3, max_value = 11, step = 2)
        kernel_size_2 = hp.Int(f"kernel_size_2", min_value = 3, max_value = 11, step = 2)

        x = Conv1D(filters=128, kernel_size=kernel_size_0, padding='same')(input_trace)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv1D(filters=256, kernel_size=kernel_size_1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv1D(filters=128, kernel_size=kernel_size_2, padding='same')(x)
        x = BatchNormalization()(x)
        fcn_output = ReLU()(x)

        # FCN output goes into Global Average Pooling
        gap = GlobalAveragePooling1D()(fcn_output)

        # LSTM Block
        # Reshape for LSTM input
        x = Reshape((QUIPU_LEN_CUT, 1))(input_trace)  # Adjust based on actual input requirement for LSTM
        x = LSTM(hp.Int('lstm_units', min_value=32, max_value=256, step=32))(x)
        lstm_output = Dropout(hp.Float('lstm_dropout', min_value=0.0, max_value=0.5, step=0.1))(x)

        # Concatenate FCN and LSTM outputs
        concatenated = concatenate([gap, lstm_output])

        # Output layer
        output = Dense(QUIPU_N_LABELS, activation='softmax', name='output')(concatenated)

        model = Model(inputs=input_trace, outputs=output)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        return model
    
    
    def fit(self, hp, model, training_function, **kwargs):
        train_acc, valid_acc, test_acc, n_epoch = training_function(hp, model)
        return {'Accuracy': test_acc}
    
class transformerHyperModel(keras_tuner.HyperModel): #Something's wrong, will fix later

    def build(self, hp):
        input_trace = Input(shape=(QUIPU_LEN_CUT,1), dtype='float32', name='input')

        # Hyperparameters
        head_size = hp.Int('head_size', min_value=32, max_value=128, step=32)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        num_transformer_blocks = 2#hp.Int('num_transformer_blocks', min_value=1, max_value=4, step=1)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

        # Transformer blocks
        x = input_trace
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout=dropout_rate)

        x = layers.GlobalAveragePooling1D()(x)

        output = Dense(QUIPU_N_LABELS, activation='softmax', name='output')(x)
        model = Model(inputs=input_trace, outputs=output)

        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        return model
    
    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    
    def fit(self, hp, model, training_function, **kwargs):
        train_acc, valid_acc, test_acc, n_epoch = training_function(hp, model)
        return {'Accuracy': test_acc}
    



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

