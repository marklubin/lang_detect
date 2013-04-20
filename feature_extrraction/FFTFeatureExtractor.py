"""
FFTFeatureExtractor.py
Provides feature representation based on FFT of provided waveform

Mark Lubin
"""
from numpy import fft

import FeatureExtractor

class FFTFeatureExtractor:


	def __init__(self,bandwidth,sample_rate):
		self.bandwidth = bandwidth
		self.sample_rate = sample_rate

	def extract(self,filename): pass
