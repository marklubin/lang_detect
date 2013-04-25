"""
extract.py
-append the feature matrices with result of running extactor on directory
-call as 
	python extract.py <DIRECTORY> <LABEL> <MAT_FILE>
	<LABEL> = language name eg. English

Mark Lubin
"""

from Detector.FeatureExtraction.smile_extractor import parse_features
from Detector.configuration import *
from scipy.io import loadmat,savemat
import numpy as np
from sys import argv
from os import path,listdir

def main():
	directory = path.join(AUDIO_DIR,argv[1])
	lang = argv[2]
	try:
		label = LANGS.index(lang)
	except Exception:
		print "No class for language %s, update configuration.py to add." % lang
		return

	outfilename = path.join(FEATURES_DIR,argv[3])
	
	try:
		 data = loadmat(outfilename)
		 X = data['X'].tolist()
		 y = [x[0] for x in data['y'].tolist()]
	except IOError:
		 X = []
		 y = []

	for fname in listdir(directory):
		features = parse_features(path.join(directory, fname))
		X.append(features)
		y.append(label)
		
	results = {'X' : np.array(X),'y' :y}

	savemat(outfilename,results)

if __name__ == '__main__':main()
