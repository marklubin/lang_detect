"""
demo.py
automatically extract features and return class

USAGE: python demo.py <THETA_WEIGHTS_MAT_FILE>

Mark Lubin
"""
from Detector.FeatureExtraction.smile_extractor import parse_features
from Detector.NeuralNetwork.NeuralNetwork import NeuralNetwork as NN
from Detector.configuration import *
from scipy.io import loadmat
from sys import argv
from os import system,unlink,environ,path
import numpy as np
import subprocess as sp


def main():
	if len(argv) != 2:
		print ("Need weights file.")
		return 

	weights_file = path.join(CLASSIFIERS_DIR,argv[1])
	fcall = "SMILExtractPA -C %s -sampleRate 44100 -channels 1 -O %s -noconsoleoutput -l 0"\
		%(path.join(CONF_BASE,CONF_RECORD_FILE),WAVE_FILE)
	fcall = fcall.split(' ')
	
	#record audio 
	print "\n\nPress any key to start recording...end recording with CTRL-C in new window."
	raw_input()
	rec_proc = sp.Popen(fcall,creationflags=(sp.CREATE_NEW_CONSOLE))

	rec_proc.wait()

	#get the features
	X = np.array(parse_features(WAVE_FILE))
	X = np.reshape(X,(1,X.shape[0]))

	#load the Neural Network weights
	theta = []
	data = loadmat(weights_file)
	nThetas = data[LAYERS_KEY]
	for i in range(0,nThetas):
		theta_name = 'T%d' % i
		theta.append(data[theta_name])

	#run the classifier
	N = NN(theta=theta)
	prediction = N.predict(X)[0]

	#get language and print result
	language = LANGS[prediction]
	print "That sounds like %s to me." % language

if __name__ == '__main__':main()