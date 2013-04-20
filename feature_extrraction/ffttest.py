#testing out the FFT and some plots
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
import numpy as np

fs,A =wavread("C:\Users\Mark\Desktop\eng\\b0095.wav")
ft = 1.0/fs #time per sample
n = len(A) # n samples
print n,ft
T = np.linspace(0,ft*n,n) #time vector
"""
plt.plot(T,A)#time domain plot
plt.show()
"""

Y = np.fft.fft(A)
Y = Y / len(Y)
f = fs/2 * np.linspace(0,1,len(Y)/2)
plt.plot(f, 2 *abs(Y[0:len(Y)/2]))
plt.show()

"""
plt.plot(A)
plt.title("Time domain of English Speaker Sample")
plt.show()
W = np.fft.fft(A)
f = np.fft.fftfreq(len(W))
plt.plot(f,W)
plt.show()

"""