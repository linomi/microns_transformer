from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import gc
from tensorflow import keras
from keras import layers
from keras.models import Model


class Tublet_projection(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
    def build(self,input_shape):

        self.projection = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(input_shape[1],-1, self.embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
        
class ModulationEmbedding(layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        _, num_tokens,embed_dim= input_shape
        self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim= embed_dim)
        self.speed_embedding = layers.Embedding(input_dim = num_tokens, output_dim= embed_dim)
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)
    def call(self, encoded_tokens,runing_speed):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_speed = self.speed_embedding(runing_speed)
        encoded_tokens = encoded_tokens + encoded_speed + encoded_positions
        return encoded_tokens

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _,_, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def build_model(num_heads:int,
                spatial_layers:int,temporal_layers:int,
                delay:int,embed_dim:int,input_shape:tuple,output_shape:tuple)->keras.models.Model:
    


    delay = delay
    num_heads =num_heads
    num_spatial_tranformers = spatial_layers
    num_temporal_transfomers = temporal_layers
    embed_dim = embed_dim
    input_shape = (delay,input_shape[1],input_shape[2],input_shape[3])
    LAYER_NORM_EPS = 1e-6
    output_shape = output_shape[1]



    inputs = layers.Input(shape=input_shape,name="image_input")
    running_speed_input = layers.Input(shape = (delay),name= "running_input")


    
    


    



    # Create patches.
    patches = Tublet_projection(patch_size=(16,16),embed_dim=embed_dim)(inputs)
    #pathces = Lambda(lambda x : x/255.0)(patches)
    #Encode patches.
    encoded_patches = PositionalEncoder(embed_dim=embed_dim)(patches)

    #spatial Encoder
    for _ in range(num_spatial_tranformers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1,attention_axes=-2
        )(x1, x1)  # attention axis = -2 indicates that attention score calculate along spatial axis

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
        #encoded_patches = layers.Dropout(rate = droputs_rate)(encoded_patches)
    # encode pathces with frame numbers and runnig speed 
    encoded_patches = layers.Reshape((delay,-1))(encoded_patches)
    encoded_patches = ModulationEmbedding()(encoded_patches,running_speed_input)
    encoded_patches = layers.Dense(units=embed_dim)(encoded_patches) # reshaping by linear transfomation into orginal embedding size

    # Temporal attention 
    for _ in range(num_temporal_transfomers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
        #encoded_patches = layers.Dropout(rate = droputs_rate)(encoded_patches)

    ## core model construction and load 
    #core_model = Model(inputs=[inputs,running_speed_input],outputs = encoded_patches)
    #if pretrained:
    #    core_model.load_weights('core_path')
    



    # feed back network 
    feedback_input = layers.Input((delay,output_shape))
    feedback = layers.Flatten()(feedback_input)
    feedback = layers.Dense(128,activation='elu')(feedback)
    feedback = layers.Dense(64,activation='elu')(feedback)
    feedback = layers.Dense(32,activation='elu')(feedback)
    feedback = layers.Dense(delay*embed_dim,activation='elu')(feedback)

    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    representation = layers.Flatten()(representation)
    #regularization = tf.keras.regularizers.L1L2(l1=regularization_term, l2=regularization_term)
    #outputs = layers.Dense(units=output_shape, activation="linear",kernel_regularizer=regularization)(representation)
    representation = layers.Concatenate()([representation,feedback])
    outputs = layers.Dense(units=output_shape, activation="linear")(representation)
    model = keras.Model(inputs=[inputs,running_speed_input,feedback_input], outputs=outputs)
    return model #,core_model
