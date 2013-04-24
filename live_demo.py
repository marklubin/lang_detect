"""
live_demo.py
automatically extract features and return class

USAGE: python live_demo.py <THETA_WEIGHTS_MAT_FILE>

Mark Lubin
"""
from feature_extraction.smile_extractor import parse_features
from neural_network.NeuralNetwork import NeuralNetwork as NN
from scipy.io import loadmat
from sys import argv
from os import system,unlink,environ
import numpy as np
import subprocess as sp

LANGS = ["English","Russian"]
CONF_BASE = environ['SMILE_CONF'] + "\\"
CONF_RECORD_FILE = "demo\\audiorecorder.conf"
WAVE_FILE = "out.wav"

def main():
	if len(argv) != 2:
		print ("Need weights file.")
		return 

	weights_file = argv[1]
	fcall = "SMILExtractPA -C %s -sampleRate 44100 -channels 1 -O %s -noconsoleoutput -l 0"\
		%(CONF_BASE + CONF_RECORD_FILE,WAVE_FILE)
	fcall = fcall.split(' ')
	
	#record audio 
	print "\n\nPress any key to start recording...end recording with CTRL-C in new window."
	raw_input()
	rec_proc = sp.Popen(fcall,creationflags=(sp.CREATE_NEW_CONSOLE))

	rec_proc.wait()

	#get the features
	X = parse_features(WAVE_FILE)

	#load the Neural Network weights
	theta = []
	data = loadmat(weights_file)
	for mat_name in data:
		theta.append(data[mat_name])

	#run the classifier
	N = NN(theta=theta)
	prediction = N.predict(X)[0]

	#get language and print result
	language = LANG[prediction]
	print "That sounds like %s to me." % language

if __name__ == '__main__':main()