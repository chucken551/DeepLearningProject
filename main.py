import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers, datasets
from tensorflow.keras import optimizers
import time
import _pickle as pickle

def normalize_data(data):
    return data / 255.0

def layer_model(in_shape, num_classes, act='relu'):
    l = Input(name='inputs', shape = in_shape, dtype = 'float32')
    k_layers = l

     # Block 1
    k_layers =Conv2D(32, (3, 3), activation=act, padding='same', strides=(1,1), name='block1_conv', input_shape=in_shape)(k_layers)

    # Flatten
    k_layers =Flatten(name='flatten')(k_layers)

    # Dense layer
    k_layers =Dense(64, activation=act, name='dense')(k_layers)

    # Predictions
    k_layers =Dense(num_classes, activation='softmax', name='pred')(k_layers)

    # Print network summary
    Model(inputs=l, outputs=k_layers).summary()

    return Model(inputs=l, outputs=k_layers)

def main():
    print('-------Loading and normalizing data-------')
    (train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)
    print(train_labels[0])
    print('-----------------Complete-----------------')
    print('--------------Creating model--------------')
    adam = optimizers.Adam(lr=0.0001)
    model = layer_model(train_data[0].shape, 10)
    model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    print('-----------------Complete-----------------')
    history = model.fit(
    train_data,
    train_labels,
    batch_size = 100,
    epochs = 50,
    callbacks = None,
    shuffle = True,
    validation_split = 0.1
)




main()
