#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:03:20 2024

@author: christian_evelynnguyen
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Setting memory growth for GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Function to create the data generators
def create_data_generator(directory, image_size=(256, 256), batch_size=32):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred", label_mode='categorical',
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42
    )

# Loading the training and test datasets
ds_train = create_data_generator('/Users/christian_evelynnguyen/Downloads/Computer Applications of Biomedical Engineering/chest_xray/train')
ds_test = create_data_generator('/Users/christian_evelynnguyen/Downloads/Computer Applications of Biomedical Engineering/chest_xray/test')



# Building a sequential model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Model summary
model.summary()

# Compiling the model
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)


history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=10,
    verbose=2,
)

# Evaluating the model
model.evaluate(ds_test, verbose=2)

# Plotting accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
