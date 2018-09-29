# -*- coding: utf-8 -*-
"""
-Naim Sen-
Jun 18
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from Learning.learn_audio_sklearn import prepare_data
from Learning.learn_audio_sklearn import plot_probability_matrix
from Learning.learn_audio_sklearn import plot_proba_std_matrix
from Learning.learn_audio_sklearn import plot_roc_curve, plot_confusion_matrix
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


"""
A function to plot accuracy and loss of training and
validation data as a function of epoch. Unlike functions
from the Learning module, it returns a figure rather than
supporting figure pass-through since these graphs should only
be plotted on a single figure
"""


def plot_nn_history(fit_history):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    fig.add_subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return fig


def run_rfc(train_perc, all_features, all_labels):
    features_train, features_test, artists_train, artists_test = train_test_split(
        all_features, all_labels, train_size=train_perc, random_state=0, stratify=all_labels)

    
    # Build a forest and compute the feature importances
    n_estimators = 2000  # number of trees?
    forest = RandomForestClassifier(
        n_estimators=n_estimators, class_weight='balanced')
    forest.fit(features_train, artists_train)
    artists_pred = forest.predict(features_test)
    artists_proba = forest.predict_proba(features_test)
    # we'll print this later as a comparison
    accuracy = accuracy_score(artists_test, artists_pred)
    """
    # fit the model
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.10)

    # prediction - get probabilities and guesses from test data
    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)
    # get test labels and prediction labels back using label encoder
    y_test_labels = label_encoder.inverse_transform(y_test.argmax(axis=-1))
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    # analytics time!
    # declare multi-plot figure
    # fig = plt.figure()
    # plot_probability_matrix(y_test_labels, y_prob, figure=fig)
    # plot_confusion_matrix(y_test_labels, y_pred_labels, figure=fig)
    # plot_proba_std_matrix(y_test_labels, y_prob, figure=fig, subplot_indices=223)
    # plot_roc_curve(y_test_labels, y_prob, figure=fig)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    
    print('number of features : {0}'.format(n_features))
    print('accuracy = {0}'.format(accuracy))
    print('--'*20)
    print(classification_report(y_test_labels, y_pred_labels))
    print('--'*20)
    print('Model summary')
    print(model.summary())
    
    history_fig = plot_nn_history(history)
    plt.figure(history_fig.number)
    plt.show()
    """

    return accuracy
    
# =================== main ======================


if __name__ == '__main__':
    
    # Set matplotlib params
    # This changes the size of created figures (including saving)
    # adjust for your display if necessary
    matplotlib.rcParams['figure.figsize'] = [18, 9]
    
    # load in previously saved data we will use similar calling idioms
    # to the sklearn version.
    path = '/raid/scratch/sen/adverts2/more_features/'
    
    all_data = glob.glob(path + '/*_data.pkl')
    # first we want to get all our data loaded in
    all_features, all_labels, feature_names = prepare_data(all_data, path)
    n_labels = len(np.unique(all_labels))
    n_features = len(feature_names)
    n_samples = len(all_labels)

    print(feature_names)
    """
    # we need to turn our string vector of class labels into ints
    # use sklearn preprocessing
    label_encoder = preprocessing.LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    # now transform to one-hot (could use tf if we want but it makes a tensor)
    ## all_labels = tf.one_hot(all_labels, n_labels)
    all_labels = keras.utils.to_categorical(all_labels)
    all_features = np.array(all_features)
    
    acc_vals = np.zeros((18, 5))
    for i in range(18):
        train_perc = (i+1)/20
        repeats = np.zeros((5))
        for j in range(5):
            repeats[j] = run_rfc(train_perc, all_features, all_labels)
            print(repeats)
        
        train_indices = list(range(5))
        pool = ThreadPool(5)
        repeats = pool.starmap(run_rfc, zip(train_indices, itertools.repeat(all_features), itertools.repeat(all_labels)))
        pool.close()
        pool.join()
        acc_vals[i] = repeats
        
    print(acc_vals)
    """
    acc_vals = [[0.29136842, 0.29515789, 0.29515789, 0.28884211, 0.29094737],
                [0.35155556, 0.35066667, 0.34933333, 0.35333333, 0.35288889],
                [0.38541176, 0.38541176, 0.384,      0.38682353, 0.39011765],
                [0.411,  0.4045, 0.4145, 0.4065, 0.4055],
                [0.44373333, 0.43893333, 0.43306667, 0.4368,     0.43466667],
                [0.45142857, 0.45257143, 0.45085714, 0.44857143, 0.44285714],
                [0.45046154, 0.46092308, 0.45661538, 0.45907692, 0.45538462],
                [0.45533333, 0.46,       0.468,      0.46466667, 0.46      ],
                [0.47272727, 0.47636364, 0.47636364, 0.47054545, 0.48218182],
                [0.4896, 0.4784, 0.4824, 0.476,  0.4744],
                [0.48088889, 0.50044444, 0.49333333, 0.47733333, 0.48977778],
                [0.491, 0.49,  0.498, 0.493, 0.5  ],
                [0.48,       0.48685714, 0.47885714, 0.46742857, 0.488     ],
                [0.49866667, 0.50933333, 0.50133333, 0.504,      0.49066667],
                [0.5184, 0.5104, 0.5104, 0.512,  0.5152],
                [0.538, 0.526, 0.514, 0.516, 0.522],
                [0.51733333, 0.52533333, 0.512,      0.50933333, 0.50933333],
                [0.52,  0.524, 0.528, 0.528, 0.516]]
    
    mean = np.mean(acc_vals, axis=1)
    stdev = np.std(acc_vals, axis=1)
    x = np.linspace(0.05, 0.9, len(mean))

    plt.figure()
    plt.errorbar(x, mean, yerr=stdev)
    plt.xlabel('Train fraction')
    plt.ylabel('Accuracy')
    #plt.xticks(np.linspace(.1, .9, 9))
    plt.grid()
    plt.title('Accuracy against training fraction for 2500 samples, 419 features, neural network')

    plt.show()    
