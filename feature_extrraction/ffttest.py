#testing out the FFT and some plots
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
import numpy as np

rate,A =wavread("C:\Users\Mark\Desktop\eng\\b0095.wav")
"""
plt.plot(A)
plt.title("Time domain of English Speaker Sample")
plt.show()
"""
W = np.fft.fft(A)
print W
plt.plot(W.real)
plt.show()