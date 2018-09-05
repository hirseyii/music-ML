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
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from Learning.learn_audio_sklearn import prepare_data
from Learning.learn_audio_sklearn import plot_probability_matrix
from Learning.learn_audio_sklearn import plot_proba_std_matrix
from Learning.learn_audio_sklearn import plot_roc_curve, plot_confusion_matrix


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


def run_nn(train_perc, all_features, all_labels, label_encoder):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(all_features,
                                                        all_labels,
                                                        train_size=train_perc,
                                                        random_state = 0,
                                                        stratify = all_labels)
    # scale data
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # number of neurons in hidden layer
    # scaling factor (larger is less prone to overfitting?)
    alpha = 2
    n_hidden = x_train.shape[0]/(alpha * (n_features + n_labels))
    if n_hidden < 2:
        n_hidden = 2

    n_hidden = 7
    
    # print('n_hidden = ', n_hidden)
    # Construct keras sequential model
    model = keras.Sequential()
    # input layer
    model.add(keras.layers.Dense(n_features, activation='sigmoid'))
    # hidden layers
    model.add(keras.layers.Dense(n_hidden, activation='sigmoid'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(n_hidden, activation='sigmoid'))
    # output layer
    model.add(keras.layers.Dense(n_labels, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy',
                  # loss='kullback_leibler_divergence',
                  optimizer='Adam',
                  metrics=['accuracy'])

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
    """
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
    """
    # load in previously saved data we will use similar calling idioms
    # to the sklearn version.
    path = '/raid/scratch/sen/adverts2/data_lib/'
    
    all_data = glob.glob(path + '/*_data.pkl')
    # first we want to get all our data loaded in
    all_features, all_labels, feature_names = prepare_data(all_data, path)
    n_labels = len(np.unique(all_labels))
    n_features = len(feature_names)
    n_samples = len(all_labels)

    print(feature_names)
    
    # we need to turn our string vector of class labels into ints
    # use sklearn preprocessing
    label_encoder = preprocessing.LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    # now transform to one-hot (could use tf if we want but it makes a tensor)
    ## all_labels = tf.one_hot(all_labels, n_labels)
    all_labels = keras.utils.to_categorical(all_labels)
    all_features = np.array(all_features)
    

    
    acc_vals = np.zeros((16, 4))
    for i in range(16):
        train_perc = (i+1)/20
        repeats = np.zeros((4))
        for j in range(4):
            repeats[j] = run_nn(train_perc, all_features, all_labels, label_encoder)
            print(repeats)
            
        acc_vals[i] = repeats
        
    print(acc_vals)
    
    repeats = np.zeros((4))
    for j in range(4):
        repeats[j] = run_nn(0.95, all_features, all_labels, label_encoder)
        print(repeats)
    """
    acc_vals = np.matrix([[0.27115789, 0.27115789, 0.28757895, 0.27831579],
                          [0.35288889, 0.32444444, 0.31333333, 0.332],
                          [0.34117647, 0.33411765, 0.33835294, 0.32329412],
                          [0.364, 0.3645, 0.386, 0.354],
                          [0.39306667, 0.38773333, 0.392, 0.3792],
                          [0.39828571, 0.40228571, 0.38685714, 0.39714286],
                          [0.43138462, 0.40738462, 0.416, 0.41661538],
                          [0.42866667, 0.41733333, 0.43066667, 0.42333333],
                          [0.424, 0.43054545, 0.43490909, 0.448],
                          [0.4632, 0.4544, 0.4416, 0.452],
                          [0.46666667, 0.46133333, 0.45155556, 0.45866667],
                          [0.474, 0.465, 0.468, 0.477],
                          [0.47428571, 0.49485714, 0.47085714, 0.48342857],
                          [0.48133333, 0.492, 0.484, 0.48533333],
                          [0.5424, 0.5408, 0.5168, 0.5312],
                          [0.542, 0.548, 0.528, 0.534],
                          [0.568, 0.56, 0.53066667, 0.552],
                          [0.576, 0.58, 0.572, 0.564],
                          [0.56, 0.584, 0.616, 0.6]])
                          
    mean = np.mean(acc_vals, axis=1)
    stdev = np.std(acc_vals, axis=1)
    x = np.linspace(0.05, 0.95, len(mean))

    plt.figure()
    plt.errorbar(x, mean, yerr=stdev)
    plt.xlabel('Train fraction')
    plt.ylabel('Accuracy')
    plt.xticks(np.linspace(.1, .9, 9))
    plt.grid()
    plt.title('Accuracy against training fraction for 2500 samples, 419 features, neural network')


    plt.show()
    
