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

# generating a softmask
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(mag_filter,
                               margin_i * (mag - mag_filter),
                               power=power)

mask_v = librosa.util.softmask(mag - mag_filter,
                               margin_v * mag_filter,
                               power=power)

front = mag * mask_v
back = mag * mask_i

lb.output.write_wav('front_softmask.wav', lb.core.istft(front), sr)
lb.output.write_wav('back_softmask.wav', lb.core.istft(back), sr)
