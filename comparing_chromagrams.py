"""
Naim Sen
-Jun 18-
"""
# Compare the chromagram of Bach and RHCP.
# The Bach was chosen to be extremely chromatic whereas Scar Tissue stays in
# the same key throughout and remains harmonic rather than chromatic.

import numpy as np
import librosa.display
import librosa as lb
import matplotlib.pyplot as plt

# load tracks
b_y, b_sr = lb.load('sample/Bach Chromatic Fantasy and Fugue in D Minor.mp3')
r_y, r_sr = lb.load('sample/03 Scar Tissue.mp3')

# compute chromagram
bach = lb.feature.chroma_cqt(y=b_y, sr=b_sr)
rhcp = lb.feature.chroma_cqt(y=r_y, sr=r_sr)


# we want to beat sync for better comparison
# get beat locs
b_tempo, b_beat = lb.beat.beat_track(y = b_y, sr=b_sr, trim=False)
r_tempo, r_beat = lb.beat.beat_track(y = r_y, sr=r_sr, trim=False)
# fix frames to fit each track
b_beat = lb.util.fix_frames(b_beat, x_max=bach.shape[1])
r_beat = lb.util.fix_frames(r_beat, x_max=rhcp.shape[1])
# sync chromagram to beats
bach_sync = lb.util.sync(bach, b_beat, aggregate=np.median)
rhcp_sync = lb.util.sync(rhcp, r_beat, aggregate=np.median)
# convert from frames to time to generate x axes
b_beat_t = lb.frames_to_time(b_beat, sr=b_sr)
r_beat_t = lb.frames_to_time(r_beat, sr=r_sr)

# display plots
bachplot = plt.subplot(2,1,1)
lb.display.specshow(bach_sync, y_axis='chroma', x_axis='time', x_coords=b_beat_t)
plt.title('Bach Chromatic Fantasy - Chroma')
plt.tight_layout()

rhcpplot = plt.subplot(2,1,2)
lb.display.specshow(rhcp_sync, y_axis='chroma', x_axis='time', x_coords=r_beat_t)
plt.title('Scar Tissue - Chroma')
plt.tight_layout()
plt.show()
