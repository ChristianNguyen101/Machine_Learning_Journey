#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:07:23 2024

@author: christian_evelynnguyen
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# downloads the MNIST Data Set
from tensorflow.keras.datasets import mnist
#allows graphing functionalities
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=keras.Sequential()
model.add(keras.Input(shape=(None,28)))
# We don't need to have a specific number of time step but we need 28 pixels per timestep
model.add(keras.layers.LSTM(256, return_sequences=True, activation='tanh'))
    # 512 nodes and returns output from each timestep
model.add(keras.layers.LSTM(256, activation='tanh'))
    # don't need return sequences = True for the second layer 
model.add(keras.layers.Dense (10)) 
print(model.summary())
# Gives Data regarding the model
model.compile(
    #labels are just integer for correct label
    # goes through losses first, then sparse categorical 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=["accuracy"],
    )
#verbose = 2 prints every time
history=model.fit(x_train,y_train,batch_size=32,epochs=10, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=32,verbose=2)


plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')