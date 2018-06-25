"""
Junk script for testing things!
"""

import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import numpy as np

# prepare track
track_path = 'sample/03 Scar Tissue.mp3'
y, sr = lb.load(track_path)

# harmonic + percussive
p_y, h_y = lb.effects.hpss(y)


# tonnetz
tonnetz = lb.feature.tonnetz(y=y, sr=sr)
tonnetz_p = lb.feature.tonnetz(y=p_y, sr=sr)
tonnetz_h = lb.feature.tonnetz(y=h_y, sr=sr)

original_plt = plt.subplot(3,1,1)
lb.display.specshow(tonnetz, y_axis='tonnetz')
plt.colorbar()

harmonic_plt = plt.subplot(3,1,2)
lb.display.specshow(tonnetz_h, y_axis='tonnetz')
plt.colorbar()

percussive_plt = plt.subplot(3,1,3)
lb.display.specshow(tonnetz_p, y_axis='tonnetz')
plt.colorbar()
plt.tight_layout()
plt.show()
