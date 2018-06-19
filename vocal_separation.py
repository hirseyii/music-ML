"""
Naim Sen
-Jun 18-
"""

# Testing out vocal separation techniques

import numpy as np
import librosa.display
import librosa as lb
import matplotlib.pyplot as plt

# import a track
track_path = 'sample/03 Scar Tissue.mp3'
y, sr = lb.load(track_path)

# separate vocals using REPET and nearest neighbour filtering
# compute spectrogram mag & phase
mag, phase = lb.magphase(lb.stft(y))
mag_filter = lb.decompose.nn_filter(mag, aggregate=np.median, metric='cosine',
                                    width=int(lb.time_to_frames(2, sr=sr)))
# select the filter such that it is always less than the combined
# (vocal + instrumental) magnitudes: pointwise minimum
mag_filter = np.minimum(mag, mag_filter)

# lets try by using the raw filter instead of a softmask
front = mag * mag_filter
back = mag * (mag - mag_filter)

lb.output.write_wav('front_hard_filter.wav', lb.core.istft(front), sr)
lb.output.write_wav('back_hard_filter.wav', lb.core.istft(back), sr)
