#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import json
import keras
from keras import layers
from keras import backend as K
import tensorflow as tf
from talos.utils.exceptions import TalosParamsError
from time import time
import pandas as pd
import numpy as np
import logging
import itertools
import gc
from numpy.random import seed
from tensorflow import set_random_seed
from scipy.stats import pearsonr


# random seed (for KFold, numpy and tensorflow)
my_seed = 2208
seed(my_seed)
set_random_seed(my_seed)


# helper function called in the fit_neural_net function
def get_hidden_shape(params, out_neurons):

    shape_type = params['shape']
    hidden_layers = params['hidden_layers']
    first_neuron = params['first_neuron']
    last_neuron = out_neurons

    if shape_type == 'brick':
        out = [first_neuron]*hidden_layers

    elif shape_type == 'triangle':
        out = np.linspace(first_neuron,
                          last_neuron,
                          hidden_layers + 2,
                          dtype=int).tolist()
        out.pop(0)
        out.pop(-1)

    else:
        raise TalosParamsError("no valid shape parameter provided")

    return out


# helper function called in the fit_neural_net function
def add_hidden_layers(model, params, layer_sizes, batch_norm=False):

    hidden_layers = params['hidden_layers'] - 1    # one hidden layer was already added in main method
    layer_sizes = layer_sizes[1:]           # remove that first hidden layer

    # gets set
    try:
        kernel_initializer = params['kernel_initializer']
    except KeyError:
        kernel_initializer = 'glorot_uniform'

    # always given
    try:
        kernel_regularizer = params['kernel_regularizer']
    except KeyError:
        kernel_regularizer = None

    # always given
    try:
        bias_initializer = keras.initializers.Constant(params['bias_initializer'])
    except KeyError:
        bias_initializer = 'zeros'

    # fine: should be none
    try:
        bias_regularizer = params['bias_regularizer']
    except KeyError:
        bias_regularizer = None

    for i in range(hidden_layers):

        if batch_norm:
            model.add(layers.Dense(layer_sizes[i],
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer, use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(params['activation']))
        else:
            model.add(layers.Dense(layer_sizes[i],
                                   activation=params['activation'],
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer))
        model.add(layers.Dropout(params['dropout']))


# helper function called in the fit_model function
def fit_neural_net(x_train, y_train, params):

    # set initialization depending on used activation
    if params['activation'] == 'tanh' or params['activation'] == 'sigmoid':
        params['kernel_initializer'] = 'glorot_uniform'
    else:
        params['kernel_initializer'] = 'he_normal'

    # determine the number of nodes in each hidden layer
    layer_sizes = get_hidden_shape(params, out_neurons=1)

    model = keras.Sequential()

    # add first hidden layer including input layer
    model.add(layers.Dense(layer_sizes[0], activation=params['activation'],
                           kernel_initializer=params['kernel_initializer'],
                           bias_initializer=keras.initializers.Constant(params['bias_initializer']),
                           kernel_regularizer=params['kernel_regularizer'], input_dim=len(x_train.keys())))

    model.add(layers.Dropout(params['dropout']))

    # add hidden layers
    add_hidden_layers(model, params, layer_sizes, batch_norm=False)

    # add output layer
    model.add(layers.Dense(1, activation='linear', kernel_initializer=params['kernel_initializer'],
                           bias_initializer=keras.initializers.Constant(params['bias_initializer'])
                           ))
    # compile model
    # opt = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer']))  # if lr should be changed
    model.compile(loss=params['losses'], optimizer=params['optimizer'], metrics=['mean_squared_error'])

    # save plots for tensorboard
    # if use_tensorboard:
    #     direct = nf.get_talos_path(tensorboard_path, params)
    #     tensorboard = TensorBoard(log_dir=direct)
    #     callbacks = [tensorboard]

    # else:
    callbacks = []

    if params['early_stopping']:
        # https://stackoverflow.com/questions/63464944/keras-loss-and-metrics-values-do-not-match-with-same-function-in-each
        # -> val_loss includes regularization, while val_mean_squared_error does not
        # -> use val_loss to stop training, but measure final performance based on val_mean_squared_error
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=params['patience'],
                                                   verbose=0, mode='min', baseline=None, restore_best_weights=True)
        callbacks.append(early_stop)

    # plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
    # model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        verbose=0, callbacks=callbacks)

    # this gives the validation MSE of the model at the epoch with smallest validation loss
    # history.history['val_mean_squared_error'][-(params['patience'] + 1)]

    print('.', end='')  # visualize number of networks trained

    return history, model


# train: pandas DataFrame of training data where rownames are cell line names/IDs and columns are features
# test: pandas DataFrame of test data where rownames are cell line names/IDs and columns are features
# method_parameters: dictionary of network hyperparameters ("activation", "optimizer", "losses", "hidden_layers",
#  "dropout", "batch_size", "epochs", "shape", "early_stopping", "bias_initializer", "kernel_regularizer", "patience")
# shuffle_seed: random seed
# returns: model predictions for training and test data (model is only trained on training data) and runtime of training
def fit_model(train, test, method_parameters, shuffle_seed):

    train_shuffled = train.sample(frac=1, random_state=shuffle_seed)

    # generate in- and outputs for train and test data
    train_inputs = train_shuffled.drop('response', axis=1)
    test_inputs = test.drop('response', axis=1)
    train_outputs = train_shuffled[['response']]
    test_outputs = test[['response']]

    start_time = time()
    history, model = fit_neural_net(train_inputs, train_outputs, method_parameters)
    end_time = time()

    # compute predictions
    predictions_train = model.predict(train_inputs, verbose=0, batch_size=train_inputs.shape[0])
    predictions_test = model.predict(test_inputs, verbose=0, batch_size=test_inputs.shape[0])

    # formatting
    predictions_test = pd.DataFrame(predictions_test)
    predictions_train = pd.DataFrame(predictions_train)
    predictions_test.columns = ["predicted_response"]
    predictions_train.columns = ["predicted_response"]
    predictions_test.index = test.index
    predictions_train.index = train_shuffled.index

    duration = (end_time - start_time)

    # cleanup
    del model
    gc.collect()
    K.clear_session()

    return predictions_test, predictions_train, duration


# helper function called in compute_errors and compute_errors_threshold_based
# based on the book "An Introduction to Statistical Learning: With Applications in R", page 234
# predicted: predicted ln(IC50) values for training or test cell lines
# actual: actual ln(IC50) values for training or test cell lines
# num_predictiors: number of input features the model was trained on
# returns: adjusted R squared value
def R_squared_adjusted(predicted, actual, num_predictors):

    num_samples = len(actual)

    if (num_samples - num_predictors - 1) < 1 :
        return np.nan
    else:
        RSS = sum(np.square([x1 - x2 for (x1, x2) in zip(actual, predicted)]))
        mean_actual = (sum(actual) / len(actual))
        TSS = sum(np.square([x1 - mean_actual for x1 in actual]))
        R2_adj = 1 - (((RSS / (num_samples - num_predictors - 1)) / (TSS / (num_samples - 1))))
        return R2_adj


# data: pandas DataFrame (either training or test samples) with cell line names/IDs as rownames
# and columns "response" (true ln(IC50)), "predicted_response" (predicted ln(IC50))
# num_features (optional): number of input features the model was trained on
# mean_train_IC50 (optional): mean ln(IC50) of training samples
# range_train_IC50 (optional): size of the range of ln(IC50) values in training data
# returns: various error measures
def compute_errors(data, num_features=None, mean_train_IC50=None, range_train_IC50=None):
    MSE = np.square(data["response"] - data["predicted_response"]).mean()
    Median_SE = np.square(data["response"] - data["predicted_response"]).median()
    MAE = np.abs(data["response"] - data["predicted_response"]).mean()
    PCC, PCC_pvalue = pearsonr(data['response'].tolist(), data['predicted_response'].tolist())

    # measures that require additional information on number of features + train data
    Adjusted_R2 = Baseline_MSE = Baseline_normalized_MSE = Range_normalized_MSE = None
    if num_features is not None:
        Adjusted_R2 = R_squared_adjusted(data["predicted_response"].tolist(), data["response"].tolist(), num_features)

    if mean_train_IC50 is not None:
        Baseline_MSE = np.square(data['response'] - mean_train_IC50).mean()
        Baseline_normalized_MSE = MSE / Baseline_MSE

    if range_train_IC50 is not None:
        Range_normalized_MSE = MSE / range_train_IC50

        return {
                "MSE": MSE,
                "Median_SE": Median_SE,
                "MAE": MAE,
                "PCC": PCC,
                "PCC_pvalue": PCC_pvalue,
                "Adjusted_R2": Adjusted_R2,
                "Baseline_MSE": Baseline_MSE,
                "Baseline_normalized_MSE": Baseline_normalized_MSE,
                "Range_normalized_MSE": Range_normalized_MSE
        }


# data: pandas DataFrame (either training or test samples) with cell line names/IDs as rownames
# and columns "response" (true ln(IC50)), "predicted_response" (predicted ln(IC50))
# drug_threshold: drug-specific ln(IC50) threshold to divide cell lines in sensitive (ln(IC50) < threshold) and resistant ones
# num_features (optional): number of input features the model was trained on
# mean_train_IC50 (optional): mean ln(IC50) of training samples
# range_train_IC50 (optional): size of the range of ln(IC50) values in training data
# returns: various error measures that require an ln(IC50) threshold
def compute_errors_threshold_based(data, drug_threshold, num_features=None, mean_train_IC50=None, range_train_IC50=None):

    data["class"] = (data["response"] < drug_threshold).astype(int)
    data["predicted_class"] = (data["predicted_response"] < drug_threshold).astype(int)

    data_sens = data[data["class"] == 1]
    data_res = data[data["class"] == 0]
    assert data.shape[0] == (data_sens.shape[0] + data_res.shape[0])

    data_correct = data[data['class'] == data['predicted_class']]
    data_incorrect = data[data['class'] != data['predicted_class']]
    data_TP = data[(data['predicted_class'] == 1) & (data['class'] == 1)]
    data_TN = data[(data['predicted_class'] == 0) & (data['class'] == 0)]
    data_FP = data[(data['predicted_class'] == 1) & (data['class'] == 0)]
    data_FN = data[(data['predicted_class'] == 0) & (data['class'] == 1)]

    assert data.shape[0] == (data_correct.shape[0] + data_incorrect.shape[0])
    assert data_correct.shape[0] == (data_TP.shape[0] + data_TN.shape[0])
    assert data_incorrect.shape[0] == (data_FP.shape[0] + data_FN.shape[0])
    assert (data_correct['predicted_class'] == data_correct['class']).all()

    if data_TP.shape[0] > 0:
        assert data_TP['predicted_class'].unique() == [1]
        assert data_TP['class'].unique() == [1]
    if data_TN.shape[0] > 0:
        assert data_TN['predicted_class'].unique() == [0]
        assert data_TN['class'].unique() == [0]
    if data_FP.shape[0] > 0:
        assert data_FP['predicted_class'].unique() == [1]
        assert data_FP['class'].unique() == [0]
    if data_FN.shape[0] > 0:
        assert data_FN['predicted_class'].unique() == [0]
        assert data_FN['class'].unique() == [1]

    TP = data_TP.shape[0]
    TN = data_TN.shape[0]
    FP = data_FP.shape[0]
    FN = data_FN.shape[0]
    assert (TP + TN + FP + FN) == data.shape[0]

    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

    MSE_sensitive_CLs =  np.square(data_sens['response'] - data_sens['predicted_response']).mean()
    MSE_resistant_CLs =  np.square(data_res['response'] - data_res['predicted_response']).mean()
    MSE_correctly_classified_CLs =  np.square(data_correct['response'] - data_correct['predicted_response']).mean()
    MSE_incorrectly_classified_CLs = np.square(data_incorrect['response'] - data_incorrect['predicted_response']).mean()
    MSE_TPs = np.square(data_TP['response'] - data_TP['predicted_response']).mean()
    MSE_TNs = np.square(data_TN['response'] - data_TN['predicted_response']).mean()
    MSE_FPs = np.square(data_FP['response'] - data_FP['predicted_response']).mean()
    MSE_FNs = np.square(data_FN['response'] - data_FN['predicted_response']).mean()

    Median_SE_sensitive_CLs =  np.square(data_sens['response'] - data_sens['predicted_response']).median()
    Median_SE_resistant_CLs =  np.square(data_res['response'] - data_res['predicted_response']).median()
    Median_SE_correctly_classified_CLs =  np.square(data_correct['response'] - data_correct['predicted_response']).median()
    Median_SE_incorrectly_classified_CLs = np.square(data_incorrect['response'] - data_incorrect['predicted_response']).median()
    Median_SE_TPs = np.square(data_TP['response'] - data_TP['predicted_response']).median()
    Median_SE_TNs = np.square(data_TN['response'] - data_TN['predicted_response']).median()
    Median_SE_FPs = np.square(data_FP['response'] - data_FP['predicted_response']).median()
    Median_SE_FNs = np.square(data_FN['response'] - data_FN['predicted_response']).median()

    MAE_sensitive_CLs =  np.abs(data_sens['response'] - data_sens['predicted_response']).mean()
    MAE_resistant_CLs =  np.abs(data_res['response'] - data_res['predicted_response']).mean()
    MAE_correctly_classified_CLs =  np.abs(data_correct['response'] - data_correct['predicted_response']).mean()
    MAE_incorrectly_classified_CLs = np.abs(data_incorrect['response'] - data_incorrect['predicted_response']).mean()
    MAE_TPs = np.abs(data_TP['response'] - data_TP['predicted_response']).mean()
    MAE_TNs = np.abs(data_TN['response'] - data_TN['predicted_response']).mean()
    MAE_FPs = np.abs(data_FP['response'] - data_FP['predicted_response']).mean()
    MAE_FNs = np.abs(data_FN['response'] - data_FN['predicted_response']).mean()

    if data_sens.shape[0] > 2:
        PCC_sensitive_CLs, PCC_pvalue_sensitive_CLs = pearsonr(data_sens['response'].tolist(), data_sens['predicted_response'].tolist())
    else:
        PCC_sensitive_CLs = PCC_pvalue_sensitive_CLs = np.nan

    if data_res.shape[0] > 2:
        PCC_resistant_CLs, PCC_pvalue_resistant_CLs = pearsonr(data_res['response'].tolist(), data_res['predicted_response'].tolist())

    else:
        PCC_resistant_CLs = PCC_pvalue_resistant_CLs = np.nan

    if data_correct.shape[0] > 2:
        PCC_correctly_classified_CLs, PCC_pvalue_correctly_classified_CLs = pearsonr(data_correct['response'].tolist(), data_correct['predicted_response'].tolist())

    else:
        PCC_correctly_classified_CLs = PCC_pvalue_correctly_classified_CLs = np.nan

    if data_incorrect.shape[0] > 2:
        PCC_incorrectly_classified_CLs, PCC_pvalue_incorrectly_classified_CLs = pearsonr(data_incorrect['response'].tolist(), data_incorrect['predicted_response'].tolist())
    else:
        PCC_incorrectly_classified_CLs = PCC_pvalue_incorrectly_classified_CLs = np.nan

    if TP > 2:
        PCC_TPs, PCC_pvalue_TPs = pearsonr(data_TP['response'].tolist(), data_TP['predicted_response'].tolist())
    else:
        PCC_TPs = PCC_pvalue_TPs = np.nan

    if TN > 2:
        PCC_TNs, PCC_pvalue_TNs = pearsonr(data_TN['response'].tolist(), data_TN['predicted_response'].tolist())
    else:
        PCC_TNs = PCC_pvalue_TNs = np.nan

    if FP > 2:
        PCC_FPs, PCC_pvalue_FPs = pearsonr(data_FP['response'].tolist(), data_FP['predicted_response'].tolist())
    else:
        PCC_FPs = PCC_pvalue_FPs = np.nan

    if FN > 2:
        PCC_FNs, PCC_pvalue_FNs = pearsonr(data_FN['response'].tolist(), data_FN['predicted_response'].tolist())
    else:
        PCC_FNs = PCC_pvalue_FNs = np.nan

    # measures that require additional information on number of features + train data
    Adjusted_R2_sensitive_CLs = Adjusted_R2_resistant_CLs = Adjusted_R2_correctly_classified_CLs = Adjusted_R2_incorrectly_classified_CLs = None
    Adjusted_R2_TPs = Adjusted_R2_TNs = Adjusted_R2_FPs = Adjusted_R2_FNs = None
    Baseline_normalized_MSE_sensitive_CLs = Baseline_normalized_MSE_resistant_CLs = Baseline_normalized_MSE_correctly_classified_CLs = Baseline_normalized_MSE_incorrectly_classified_CLs = None
    Baseline_normalized_MSE_TPs = Baseline_normalized_MSE_TNs = Baseline_normalized_MSE_FPs = Baseline_normalized_MSE_FNs = None
    Range_normalized_MSE_sensitive_CLs = Range_normalized_MSE_resistant_CLs = Range_normalized_MSE_correctly_classified_CLs = Range_normalized_MSE_incorrectly_classified_CLs = None
    Range_normalized_MSE_TPs = Range_normalized_MSE_TNs = Range_normalized_MSE_FPs = Range_normalized_MSE_FNs = None

    if num_features is not None:
        Adjusted_R2_sensitive_CLs = R_squared_adjusted(data_sens['predicted_response'].tolist(), data_sens['response'].tolist(), num_features)
        Adjusted_R2_resistant_CLs = R_squared_adjusted(data_res['predicted_response'].tolist(), data_res['response'].tolist(), num_features)
        Adjusted_R2_correctly_classified_CLs = R_squared_adjusted(data_correct['predicted_response'].tolist(), data_correct['response'].tolist(), num_features)
        Adjusted_R2_incorrectly_classified_CLs = R_squared_adjusted(data_incorrect['predicted_response'].tolist(), data_incorrect['response'].tolist(), num_features)
        Adjusted_R2_TPs = R_squared_adjusted(data_TP['predicted_response'].tolist(), data_TP['response'].tolist(), num_features)
        Adjusted_R2_TNs = R_squared_adjusted(data_TN['predicted_response'].tolist(), data_TN['response'].tolist(), num_features)
        Adjusted_R2_FPs = R_squared_adjusted(data_FP['predicted_response'].tolist(), data_FP['response'].tolist(), num_features)
        Adjusted_R2_FNs = R_squared_adjusted(data_FN['predicted_response'].tolist(), data_FN['response'].tolist(), num_features)

    if mean_train_IC50 is not None:
        Baseline_MSE_sensitive_CLs = np.square(data_sens['response'] - mean_train_IC50).mean()
        Baseline_MSE_resistant_CLs = np.square(data_res['response'] - mean_train_IC50).mean()
        Baseline_MSE_correctly_classified_CLs = np.square(data_correct['response'] - mean_train_IC50).mean()
        Baseline_MSE_incorrectly_classified_CLs = np.square(data_incorrect['response'] - mean_train_IC50).mean()
        Baseline_MSE_TPs = np.square(data_TP['response'] - mean_train_IC50).mean()
        Baseline_MSE_TNs = np.square(data_TN['response'] - mean_train_IC50).mean()
        Baseline_MSE_FPs = np.square(data_FP['response'] - mean_train_IC50).mean()
        Baseline_MSE_FNs = np.square(data_FN['response'] - mean_train_IC50).mean()

        Baseline_normalized_MSE_sensitive_CLs = MSE_sensitive_CLs / Baseline_MSE_sensitive_CLs
        Baseline_normalized_MSE_resistant_CLs = MSE_resistant_CLs / Baseline_MSE_resistant_CLs
        Baseline_normalized_MSE_correctly_classified_CLs = MSE_correctly_classified_CLs / Baseline_MSE_correctly_classified_CLs
        Baseline_normalized_MSE_incorrectly_classified_CLs = MSE_incorrectly_classified_CLs / Baseline_MSE_incorrectly_classified_CLs
        Baseline_normalized_MSE_TPs = MSE_TPs / Baseline_MSE_TPs
        Baseline_normalized_MSE_TNs = MSE_TNs / Baseline_MSE_TNs
        Baseline_normalized_MSE_FPs = MSE_FPs / Baseline_MSE_FPs
        Baseline_normalized_MSE_FNs = MSE_FNs / Baseline_MSE_FNs

    if range_train_IC50 is not None:
        Range_normalized_MSE_sensitive_CLs = MSE_sensitive_CLs / range_train_IC50
        Range_normalized_MSE_resistant_CLs = MSE_resistant_CLs / range_train_IC50
        Range_normalized_MSE_correctly_classified_CLs = MSE_correctly_classified_CLs / range_train_IC50
        Range_normalized_MSE_incorrectly_classified_CLs = MSE_incorrectly_classified_CLs / range_train_IC50
        Range_normalized_MSE_TPs = MSE_TPs / range_train_IC50
        Range_normalized_MSE_TNs = MSE_TNs / range_train_IC50
        Range_normalized_MSE_FPs = MSE_FPs / range_train_IC50
        Range_normalized_MSE_FNs = MSE_FNs / range_train_IC50

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Sensitivity": Sensitivity,
        "Specificity": Specificity,
        "MSE_sensitive_CLs": MSE_sensitive_CLs,
        "MSE_resistant_CLs": MSE_resistant_CLs,
        "MSE_correctly_classified_CLs": MSE_correctly_classified_CLs,
        "MSE_incorrectly_classified_CLs": MSE_incorrectly_classified_CLs,
        "MSE_TPs": MSE_TPs,
        "MSE_TNs": MSE_TNs,
        "MSE_FPs": MSE_FPs,
        "MSE_FNs": MSE_FNs,
        "Median_SE_sensitive_CLs": Median_SE_sensitive_CLs,
        "Median_SE_resistant_CLs": Median_SE_resistant_CLs,
        "Median_SE_correctly_classified_CLs": Median_SE_correctly_classified_CLs,
        "Median_SE_incorrectly_classified_CLs": Median_SE_incorrectly_classified_CLs,
        "Median_SE_TPs": Median_SE_TPs,
        "Median_SE_TNs": Median_SE_TNs,
        "Median_SE_FPs": Median_SE_FPs,
        "Median_SE_FNs": Median_SE_FNs,
        "MAE_sensitive_CLs": MAE_sensitive_CLs,
        "MAE_resistant_CLs": MAE_resistant_CLs,
        "MAE_correctly_classified_CLs": MAE_correctly_classified_CLs,
        "MAE_incorrectly_classified_CLs": MAE_incorrectly_classified_CLs,
        "MAE_TPs": MAE_TPs,
        "MAE_TNs": MAE_TNs,
        "MAE_FPs": MAE_FPs,
        "MAE_FNs": MAE_FNs,
        "PCC_sensitive_CLs": PCC_sensitive_CLs,
        "PCC_pvalue_sensitive_CLs": PCC_pvalue_sensitive_CLs,
        "PCC_resistant_CLs": PCC_resistant_CLs,
        "PCC_pvalue_resistant_CLs": PCC_pvalue_resistant_CLs,
        "PCC_correctly_classified_CLs": PCC_correctly_classified_CLs,
        "PCC_pvalue_correctly_classified_CLs": PCC_pvalue_correctly_classified_CLs,
        "PCC_incorrectly_classified_CLs": PCC_incorrectly_classified_CLs,
        "PCC_pvalue_incorrectly_classified_CLs": PCC_pvalue_incorrectly_classified_CLs,
        "PCC_TPs": PCC_TPs,
        "PCC_pvalue_TPs": PCC_pvalue_TPs,
        "PCC_TNs": PCC_TNs,
        "PCC_pvalue_TNs": PCC_pvalue_TNs,
        "PCC_FPs": PCC_FPs,
        "PCC_pvalue_FPs": PCC_pvalue_FPs,
        "PCC_FNs": PCC_FNs,
        "PCC_pvalue_FNs": PCC_pvalue_FNs,
        "Adjusted_R2_sensitive_CLs": Adjusted_R2_sensitive_CLs,
        "Adjusted_R2_resistant_CLs": Adjusted_R2_resistant_CLs,
        "Adjusted_R2_correctly_classified_CLs": Adjusted_R2_correctly_classified_CLs,
        "Adjusted_R2_incorrectly_classified_CLs": Adjusted_R2_incorrectly_classified_CLs,
        "Adjusted_R2_TPs": Adjusted_R2_TPs,
        "Adjusted_R2_TNs": Adjusted_R2_TNs,
        "Adjusted_R2_FPs": Adjusted_R2_FPs,
        "Adjusted_R2_FNs": Adjusted_R2_FNs,
        "Baseline_normalized_MSE_sensitive_CLs": Baseline_normalized_MSE_sensitive_CLs,
        "Baseline_normalized_MSE_resistant_CLs": Baseline_normalized_MSE_resistant_CLs,
        "Baseline_normalized_MSE_correctly_classified_CLs": Baseline_normalized_MSE_correctly_classified_CLs,
        "Baseline_normalized_MSE_incorrectly_classified_CLs": Baseline_normalized_MSE_incorrectly_classified_CLs,
        "Baseline_normalized_MSE_TPs": Baseline_normalized_MSE_TPs,
        "Baseline_normalized_MSE_TNs": Baseline_normalized_MSE_TNs,
        "Baseline_normalized_MSE_FPs": Baseline_normalized_MSE_FPs,
        "Baseline_normalized_MSE_FNs": Baseline_normalized_MSE_FNs,
        "Range_normalized_MSE_sensitive_CLs": Range_normalized_MSE_sensitive_CLs,
        "Range_normalized_MSE_resistant_CLs": Range_normalized_MSE_resistant_CLs,
        "Range_normalized_MSE_correctly_classified_CLs": Range_normalized_MSE_correctly_classified_CLs,
        "Range_normalized_MSE_incorrectly_classified_CLs": Range_normalized_MSE_incorrectly_classified_CLs,
        "Range_normalized_MSE_TPs": Range_normalized_MSE_TPs,
        "Range_normalized_MSE_TNs": Range_normalized_MSE_TNs,
        "Range_normalized_MSE_FPs": Range_normalized_MSE_FPs,
        "Range_normalized_MSE_FNs": Range_normalized_MSE_FNs
    }
