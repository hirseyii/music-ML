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
from Learning.learn_audio_sklearn import prepare_data



# =================== main ======================


if __name__ == '__main__':
    # Set matplotlib params
    # This changes the size of created figures (including saving)
    # adjust for your display if necessary
    matplotlib.rcParams['figure.figsize'] = [18, 9]

    # load in previously saved data we will use similar calling idioms
    # to the sklearn version.
    path = sys.argv[1]
    train_perc = float(sys.argv[2])
    all_data = glob.glob(path + '/*_data.pkl')
    # first we want to get all our data loaded in
    all_features, all_labels, feature_names = prepare_data(all_data, path)
    n_labels = len(np.unique(all_labels))
    n_features = len(feature_names)
    n_samples = len(all_labels)

    # we need to turn our string vector of class labels into ints
    # use sklearn preprocessing
    label_encoder = preprocessing.LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    # now transform to one-hot (could use tf if we want but it makes a tensor)
    ## all_labels = tf.one_hot(all_labels, n_labels)
    all_labels = keras.utils.to_categorical(all_labels)
    all_features = np.array(all_features)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(all_features,
                                                        all_labels,
                                                        train_size = train_perc,
                                                        random_state = 0,
                                                        stratify = all_labels
                                                        )
    # scale data
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Construct keras sequential model
    model = keras.Sequential()
    # input layer
    model.add(keras.layers.Dense(n_features, activation='sigmoid'))
    # hidden layers
    model.add(keras.layers.Dense((n_features+n_classes)/2, activation='sigmoid'))
    # output layer
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    # fit the model
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.10)
