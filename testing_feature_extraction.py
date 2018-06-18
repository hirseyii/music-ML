"""
Naim Sen
-Jun 18-
"""
# A script to test simple feature extraction using librosa
# Test using audio included in the sample folder (Red Hot Chilli Peppers albums)

import numpy as np
import librosa.display
import librosa as lb
import matplotlib.pyplot as plt

#select a single audio track
filename = 'sample/03 Scar Tissue.mp3'

#call lb load
track, sr = lb.load(filename)
#compute chromagram
chromagram_cqt = lb.feature.chroma_cqt(y=track, sr=sr)
# display cqt
plt.figure(figsize=(10, 4))
lb.display.specshow(chromagram_cqt)
plt.colorbar()
plt.title("cqt")
plt.show()

#--------------------
# tonnetz
tonnetz = lb.feature.tonnetz(y=track, sr=sr)
lb.display.specshow(tonnetz, y_axis='tonnetz')
plt.colorbar()
plt.show()
