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
margin_i, margin_v = 2, 15
power = 2

mask_i = lb.util.softmask(mag_filter,
                               margin_i * (mag - mag_filter),
                               power=power)

mask_v = lb.util.softmask(mag - mag_filter,
                               margin_v * mag_filter,
                               power=power)

front = mag * mask_v
back = mag * mask_i
# Write to file
lb.output.write_wav('front_softmask2.wav', lb.core.istft(front), sr)
lb.output.write_wav('back_softmask2.wav', lb.core.istft(back), sr)
"""
# generate 3 chromagrams
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
lb.display.specshow(lb.amplitude_to_db(mag, ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
lb.display.specshow(lb.amplitude_to_db(back, ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
lb.display.specshow(lb.amplitude_to_db(front, ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()
"""
