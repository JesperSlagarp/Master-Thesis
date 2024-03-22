# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:23:22 2023

@author: JK-WORK
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda, Masking, Dense, Activation, Dropout, Flatten, Input, BatchNormalization, MultiHeadAttention, LayerNormalization, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, Dense, ReLU, Add, Reshape, LSTM, concatenate
from params import QUIPU_LEN_CUT,QUIPU_N_LABELS
import numpy as np
import keras_tuner

class FCN(keras_tuner.HyperModel):
    def build(self, hp):

        input_trace = Input(shape=(None,1), dtype='float32', name='input',)
        x = input_trace

        #min_layers = 4
        max_layers = 8
        num_layers = max_layers#hp.Int("num_layers", min_layers, max_layers)

        for i in range(max_layers): #Because keras_tuner seems to have a bug
            hp.Choice(f"filters_block_{i}", [32, 64, 128, 256, 512])
            hp.Int(f"kernel_size_block_{i}", min_value=3, max_value=13, step=2)
        
        for i in range(num_layers):
            x = Conv1D(hp.get(f"filters_block_{i}"), hp.get(f"kernel_size_block_{i}"), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        x = GlobalAveragePooling1D()(x)

        output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
        model = Model(inputs=input_trace, outputs=output_barcode)

        learning_rate = 0.001#hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    
class ResNet(keras_tuner.HyperModel):

    def build(self, hp):

        num_initial_filters = hp.Choice("num_initial_filters", [16, 32, 64, 128])
        initial_conv_kernel = hp.Choice("initial_conv_kernel", [0, 5, 7, 9])
        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        num_super_blocks = hp.Int("num_super_blocks", 2, 3)
        num_blocks_per_super_block = hp.Int("num_blocks_per_super_block", 2, 4)

        input_trace = Input(shape=(None,1), dtype='float32', name='input')
        x = input_trace
        if(initial_conv_kernel > 0):
            x = Conv1D(num_initial_filters, initial_conv_kernel, padding = 'same',strides=2)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        for i in range(num_super_blocks):
            x = self.super_block(x, num_initial_filters, kernel_size, num_blocks_per_super_block, is_first_super_block = True if i == 0 else False)

        x = GlobalAveragePooling1D()(x)

        output_barcode = Dense(QUIPU_N_LABELS, activation='softmax', name='output_barcode')(x)
        model = Model(inputs=input_trace, outputs=output_barcode)

        learning_rate = 0.001#hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
    
    @staticmethod
    def super_block(input_tensor, filters, kernel_size, num_blocks, is_first_super_block = False):

        x = ResNet.residual_block(input_tensor, filters, kernel_size, halve = False if is_first_super_block else True)
        
        for i in range(num_blocks - 1):
            x = ResNet.residual_block(x, filters, kernel_size)
        
        return x
    
    @staticmethod
    def residual_block(input_tensor, filters, kernel_size, halve = False):
        
        # First component of the main path
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', strides = 2 if halve else 1)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second component
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        # Shortcut path
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same', strides = 2 if halve else 1)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        # Adding shortcut to the main path
        x = Add()([x, shortcut])
        x = ReLU()(x)
        
        return x

    
class LSTM_FCN(keras_tuner.HyperModel):

    def build(self, hp):

        input_trace = Input(shape=(None,1), dtype='float32', name='input')

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

        gap = GlobalAveragePooling1D()(fcn_output)

        # LSTM Block
        #masked_input = Masking(mask_value=0.0)(input_trace)
        y = LSTM(hp.Int('lstm_units', min_value=32, max_value=256, step=32))(input_trace)#(masked_input)
        lstm_output = Dropout(hp.Float('lstm_dropout', min_value=0.0, max_value=0.5, step=0.1))(y)

        # Concatenate FCN and LSTM outputs
        concatenated = concatenate([gap, lstm_output])

        # Output layer
        output = Dense(QUIPU_N_LABELS, activation='softmax', name='output')(concatenated)

        model = Model(inputs=input_trace, outputs=output)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

class Transformer(keras_tuner.HyperModel):

    def build(self, hp):

        MAX_SEQ_LEN = 1000
        head_size = hp.Choice('head_size', values=[32, 64, 128, 256])
        num_heads = hp.Choice('num_heads', values=[2, 4, 8])
        ff_dim = hp.Int('ff_dim', min_value=256, max_value=1024, step=128)
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=6, step=1)
        min_mlp_layers = 1
        max_mlp_layers = 5
        mlp_layers = hp.Int("mlp_layers", min_mlp_layers, max_mlp_layers)
        mlp_dropout = hp.Float('mlp_dropout', min_value=0.1, max_value=0.5, step=0.1)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        embed_dim = hp.Choice('embed_dim', values=[128, 256, 512])

        input_trace = keras.Input(shape=(None,1), dtype='float32', name='input')
        x = input_trace

        mask = Lambda(lambda x: tf.cast(tf.not_equal(x, 0), dtype='float32'))(x)
        mask = tf.squeeze(mask, -1)
        
        positional_encoding = self.get_positional_encoding(MAX_SEQ_LEN, embed_dim)
        pos_encoding_layer = Lambda(
            lambda x: positional_encoding[:tf.shape(x)[1], :],
            output_shape=lambda input_shape: (input_shape[1], embed_dim)
        )
        x += pos_encoding_layer(x)

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout, mask)

        x = GlobalAveragePooling1D(data_format="channels_last")(x)

        for i in range(max_mlp_layers):
            hp.Choice(f"mlp_dim_{i}", values=[64,128,256])

        for i in range(mlp_layers):
            x = layers.Dense(hp.get(f"mlp_dim_{i}"), activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)

        output = layers.Dense(QUIPU_N_LABELS, activation="softmax")(x)
        model = Model(inputs=input_trace, outputs=output)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
    
    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, mask=None):
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]

        # Multi-Head Attention and Normalization
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, attention_mask=attention_mask)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs  # Add & Norm

        # Position-wise Feed-Forward Network
        x = Dense(ff_dim, activation="relu")(res)  # First dense layer with ReLU
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)  # Second dense layer
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res  # Add & Norm
    
    @staticmethod
    def get_positional_encoding(max_seq_len, embed_dim):
        positional_encoding = np.array([
            [pos / np.power(10000, 2 * (j // 2) / embed_dim) for j in range(embed_dim)]
            if pos != 0 else np.zeros(embed_dim) 
            for pos in range(max_seq_len)
            ])
        positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i
        positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1
        return tf.cast(positional_encoding, dtype=tf.float32)
    
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

