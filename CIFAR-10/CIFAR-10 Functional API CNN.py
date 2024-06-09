#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:08:22 2024

@author: christian_nguyen
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
# This is an image identification dataset 
(x_train,y_train), (x_test,y_test) = cifar10.load_data()
# splits data into train and test sets
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0
#change type to float 32 and normalizes

def my_model():
    inputs = keras.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(32,3)(inputs)
    x = keras.layers.BatchNormalization()(x)
    # Batch normalization helps to stablize the network by normalizing the inputs between layers
    x = keras.activations.relu(x)
    # convolution -> normalization -> relu activation
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64,5,padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = keras.layers.Conv2D(128,3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x= keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    model=keras.Model(inputs=inputs, outputs=outputs)
    return model

model= my_model()
print (model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
   metrics=['accuracy'],
    )
history=model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test,y_test,batch_size=64, verbose=2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')