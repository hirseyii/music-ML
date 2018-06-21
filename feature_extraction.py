"""
Naim Sen
-Jun 18-
"""

# Extract, package, and display features from a track specified by a filepath

import numpy as np
import librosa.display
import librosa as lb
import matplotlib.pyplot as plt
from collections import OrderedDict

# define a function to grab spectral features from an input waveform 'y' and
# add the features to a dictionary. Requires a user-specified dictionary key
# prefix, e.g. t_ for total, p_ for percussive, h_ for harmonic, etc.
# Ensure that the prefix doesn't already exist in feature_dict
def get_spectral_features(feature_dict, prefix, y, sr, n_fft, hop_length):
    try:
        centroid = lb.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        centroid_avg = np.mean(centroid)
        centroid_std = np.std(centroid)
        features_dict.update({prefix+'centroid_avg':centroid_avg, prefix+'centroid_std':centroid_std})
        # spectral bandwidth
        bandwidth = lb.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bandwidth_avg = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        features_dict.update({prefix+'bandwidth_avg':bandwidth_avg, prefix+'bandwidth_std':bandwidth_std})
        # spectral contrast
        contrast = lb.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        contrast_avg = np.mean(contrast)
        contrast_std = np.std(contrast)
        features_dict.update({prefix+'contrast_avg':contrast_avg, prefix+'contrast_std':contrast_std})
        # spectral flatness
        flatness = lb.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        flatness_avg = np.mean(flatness)
        flatness_std = np.std(flatness)
        features_dict.update({prefix+'flatness_avg':flatness_avg, prefix+'flatness_std':flatness_std})
        # spectral rolloff
        rolloff = lb.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff_avg = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        features_dict.update({prefix+'rolloff_avg':rolloff_avg, prefix+'rolloff_std':rolloff_std})
    except Exception as ex:
        print('ERROR >> get_spectral_features() failed.\n')
        print(type(ex))
        print(ex.args)
        print(ex)
# prepare track and dictionary for storing features
track_path = 'sample/03 Scar Tissue.mp3'
n_fft = 2048
hop_length = 512

y, sr = lb.load(track_path)
features_dict = OrderedDict()

# ===============useful decompositions================
p_y, h_y = lb.effects.hpss(y)

# get whole-track features
get_spectral_features(features_dict, 't_', y, sr, n_fft, hop_length)
# get percussive/harmonic features
get_spectral_features(features_dict, 'p_', p_y, sr, n_fft, hop_length)
get_spectral_features(features_dict, 'h_', h_y, sr, n_fft, hop_length)
# --------------chromatic/tonal features--------------
# compute beat-synched chromagram
