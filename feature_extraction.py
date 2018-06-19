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

# prepare track and dictionary for storing features
track_path = 'sample/03 Scar Tissue.mp3'
y, sr = lb.load(track_path)
features_dict = OrderedDict()

# ===============whole-track features=================
# -----------------spectral features------------------
# spectral centroid
t_centroid = lb.feature.spectral_centroid(y=y, sr=sr)
t_centroid_avg = np.mean(t_centroid)
t_centroid_std = np.std(t_centroid)
features_dict.update({'t_centroid_avg':t_centroid_avg, 't_centroid_std':t_centroid_std})
# spectral bandwidth
t_bandwidth = lb.feature.spectral_bandwidth(y=y, sr=sr)
t_bandwidth_avg = np.mean(t_bandwidth)
t_bandwidth_std = np.std(t_bandwidth)
features_dict.update({'t_bandwidth_avg':t_bandwidth_avg, 't_bandwidth_std':t_bandwidth_std})
# spectral contrast
t_contrast = lb.feature.spectral_contrast(y=y, sr=sr)
t_contrast_avg = np.mean(t_contrast)
t_contrast_std = np.std(t_contrast)
features_dict.update({'t_contrast_avg':t_contrast_avg, 't_contrast_std':t_contrast_std})
# spectral flatness
t_flatness = lb.feature.spectral_flatness(y=y)
t_flatness_avg = np.mean(t_flatness)
t_flatness_std = np.std(t_flatness)
features_dict.update({'t_flatness_avg':t_flatness_avg, 't_flatness_std':t_flatness_std})
# spectral rolloff
t_rolloff = lb.feature.spectral_rolloff(y=y, sr=sr)
t_rolloff_avg = np.mean(t_rolloff)
t_rolloff_std = np.std(t_rolloff)
features_dict.update({'t_rolloff_avg':t_rolloff_avg, 't_rolloff_std':t_rolloff_std})
# ------------------chromatic features----------------
