#!/usr/bin/python

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model, callbacks
from random import randint
import numpy as np
import math
import pandas as pd
from time import time
import keras
import sys
from numpy.random import seed
from tensorflow import set_random_seed
import os
import keras.backend as K
import gc


# see descriptions below for format of required inputs
# train_data: gene expression values and IC50 for drug of interest of training cell lines
# test_data: gene expression values and IC50 for drug of interest of test cell lines
# latent_size number of nodes in bottleneck layer, i.e., number of features to be computed
# shuffle_seed: seed for shuffling training data
# returns: features computed by autoencoder (trained on training data) for training and test data
def fit_autoencoder(train_data, test_data, latent_size, shuffle_seed):

    train_shuffled = train_data.sample(frac=1, random_state=shuffle_seed)

    # two hidden layers before + after encoded layer
    hidden_size1 = math.ceil(train_shuffled.shape[1] / 5)
    hidden_size2 = math.ceil(hidden_size1 / 5)

    # encoder
    input_layer = layers.Input(shape=train_shuffled.shape[1:])
    hidden1 = layers.Dense(hidden_size1, activation='relu')(input_layer)
    hidden2 = layers.Dense(hidden_size2, activation='relu')(hidden1)
    latent = layers.Dense(latent_size)(hidden2)
    encoder = Model(inputs=input_layer, outputs=latent, name='encoder')
    # encoder.summary()

    # decoder
    input_layer_decoder = layers.Input(shape=encoder.output.shape[1:])
    hidden_decoder1 = layers.Dense(hidden_size2, activation='relu')(input_layer_decoder)
    hidden_decoder2 = layers.Dense(hidden_size1, activation='relu')(hidden_decoder1)
    upsampled = layers.Dense(train_shuffled.shape[1], activation='relu')(hidden_decoder2)
    decoder = Model(inputs=input_layer_decoder, outputs=upsampled, name='decoder')
    # decoder.summary()

    # complete autoencoder
    autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
    # autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    start_time = time()
    autoencoder.fit(train_shuffled, train_shuffled, batch_size=64, validation_split=0.2, epochs=100,
                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5,
                                                                       verbose=0, mode='min', baseline=None,
                                                                       restore_best_weights=True)], verbose=0)
    end_time = time()
    duration = (end_time - start_time)
    # print("Duration: ", str(duration))

    columns = ['F' + str(i) for i in range(1, latent_size + 1)]

    # encode train data
    train_encoded = encoder.predict(train_data)
    train_encoded = pd.DataFrame(train_encoded, columns=columns, index=train_data.index.values)
    # train_decoded = decoder.predict(train_encoded)

    # encode test data
    test_encoded = encoder.predict(test_data)
    test_encoded = pd.DataFrame(test_encoded, columns=columns, index=test_data.index.values)
    # test_decoded = decoder.predict(test_encoded)

    # cleanup
    gc.collect()
    K.clear_session()

    return train_encoded, test_encoded


print(sys.argv)
train_file = sys.argv[1] # path to file containing names/IDs of training cell lines (one per row)
test_file = sys.argv[2] # path to file containing names/IDs of test cell lines (one per row)
output_dir = sys.argv[3] # path to output folder for saving results
output_file = sys.argv[4] # name of output file
number_of_features = int(sys.argv[5]) # size of autoencoder bottleneck layer, number of features to be computed
exp_path = int(sys.argv[6]) # path to gene expression matrix, rownames are cell line names/IDs, columnnames are gene names
my_seed = int(sys.argv[7]) # random seed

seed(my_seed)
set_random_seed(my_seed)

exp_data = pd.read_csv(exp_path, sep=" ")
train_samples = pd.read_csv(train_file, sep='\t', header=None, index_col=0)
test_samples = pd.read_csv(test_file, sep='\t', header=None, index_col=0)

# combine expression data and drug response into one frame
train = exp_data.loc[train_samples.index.values]
assert set(train_samples.index.values) == set(train.index.values)
test = exp_data.loc[test_samples.index.values]
assert set(test_samples.index.values) == set(test.index.values)

# fit encoder and generate features
train_encoding, test_encoding = fit_autoencoder(train, test, number_of_features, my_seed)

# write encoded data into files
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
train_encoding.to_csv(output_dir + output_file + "_train.txt", sep="\t")
test_encoding.to_csv(output_dir + output_file + "_test.txt", sep="\t")

