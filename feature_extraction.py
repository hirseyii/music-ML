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

#----------------whole-track features------------------
# spectral features
t_centroid = lb.feature.spectral_centroid(y=y, sr=sr)
t_centroid_avg = np.mean(t_centroid)
t_centroid_std = np.std(t_centroid)

print(t_centroid_avg, t_centroid_std)
