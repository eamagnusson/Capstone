import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

segment_size = 1024

fs, x = wavfile.read('/Users/evanmagnusson/sysCapstone/Capstone/10/M4a_files/Swear_audiodata_6d2b2280ae7ba426_1603827151662.wav')
x = x / len(x)  # scale signal to [-1.0 .. 1.0]

noverlap = segment_size / 2
f, Pxx = signal.welch(x[ : , 0],                        # signal
                      fs=fs,                    # sample rate
                      nperseg=segment_size,     # segment size
                      window='hanning',         # window type to use
                      nfft=segment_size,        # num. of samples in FFT
                      detrend=False,            # remove DC part
                      scaling='spectrum',       # return power spectrum [V^2]
                      noverlap=noverlap)        # overlap between segments

# set 0 dB to energy of sine wave with maximum amplitude
ref = (1/np.sqrt(2)**2)   # simply 0.5 ;)
p = 10 * np.log10(Pxx/ref)

fill_to = -150 * (np.ones_like(p))  # anything below -150dB is irrelevant
plt.fill_between(f, p, fill_to )
# plt.xlim([f[2], f[-1]])
# plt.ylim([-90, 6])
# plt.xscale('log')   # uncomment if you want log scale on x-axis
plt.xlabel('f, Hz')
plt.ylabel('Power spectrum, dB')
plt.grid(True)
plt.show()