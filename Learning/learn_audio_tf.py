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
from Learning.learn_audio_sklearn import prepare_data


# =================== main ======================


if __name__ == '__main__':
    # Set matplotlib params
    # This changes the size of created figures (including saving)
    # adjust for your display if necessary
    matplotlib.rcParams['figure.figsize'] = [18, 9]

    # load in previously sved data
    path = sys.argv[1]
    all_data = glob.glob(path + '/*_data.pkl')
    # first we want to get all our data loaded in
    all_features, all_labels, feature_names = prepare_data(all_data, path)

    # we need to turn our string vector of class labels into ints
    # so we may as well binarize too
    all_labels = pd.get_dummies(all_labels)
    all_features = np.array(all_features)
    #  make tf dataset from labels and features
    assert all_features.shape[0] == all_labels.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))
    dataset = dataset.shuffle(len(all_labels))
    iterator = dataset.make_one_shot_iterator()
    el = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run(el))
    
