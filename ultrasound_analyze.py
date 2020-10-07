import librosa
import scipy
import numpy as np

import matplotlib.pyplot as plt
import librosa.display

# Importing the Audio Data
audio_data = '/Users/evanmagnusson/sysCapstone/Data1/Bluetooth and sound/2/3ac1e7a9c2148421/Swear_audiodata_3ac1e7a9c2148421_1597773118904.m4a'
audio, sr = librosa.load(audio_data, sr=44100)
print(type(audio), type(sr))
print(audio.shape, sr)

# Plot the Waveform (Amplitude vs. Time)
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(audio, sr=sr)
# plt.show()

# Plot the Spectrogram (Frequency+Amplitude vs. Time)
# X = librosa.stft(audio)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.colorbar()
# plt.show()

# Plotting FFT (Amplitude vs. Frequency)
n = len(audio)
T = 1/sr
yf = scipy.fft(audio)
xf = np.linspace(0.0, 1.0//(2.0*T), n//2)
fig, ax = plt.subplots()
ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()