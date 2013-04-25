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
from os import path,listdir
import argparse

def main():
	#set up argument parsers
	parser = argparse.ArgumentParser("Extract audio features from directory of files using SMILE.")
	parser.add_argument('directory',metavar='DIRECTORY',type=str,help='directory to parse, relative to audio folder.')
	parser.add_argument('-L',metavar="LANGUAGE",type=str,choices=LANGS,required=True,
		help='one of ' + str(LANGS) + " add languages in Detector/configuration.py")
	parser.add_argument('-o',metavar="OUTFILE",type=str,required=True,\
		help ='location of output file relative to features folder, appends file if exists else creates it.')
	parser.add_argument('-c',metavar="FEATURE CONFIGURATION FILE",type=str,required=False,default=CONF_FILE,\
						help="configuration file to use defaults to CONF_FILE.")
	parser.add_argument('--CALL',metavar="SMILE CALL",type=str,required=False,default=SMILE_CALL,\
						help="call to SMILExtract defaults to SMILE_CALL")

	args = parser.parse_args()
	directory = path.join(AUDIO_DIR,args.directory)
	lang = args.L
	label = LANGS.index(lang)
	outfilename = path.join(FEATURES_DIR,args.o)
	configfile = args.c
	smile_call = args.CALL
	
	try:
		 data = loadmat(outfilename)
		 X = data['X'].tolist()
		 y = [x[0] for x in data['y'].tolist()]
	except IOError:
		 X = []
		 y = []

	for fname in listdir(directory):
		features = parse_features(path.join(directory, fname),smile_call,configfile)
		X.append(features)
		y.append(label)
		

	nLabels = len(LANGS)
	results = {X_KEY : np.array(X),
			   Y_KEY :y, 
			   NLABELS_KEY : nLabels, 
			   FEATURE_SET_KEY : configfile,
			   LANGS_KEY        : '-'.join(LANGS)}

	savemat(outfilename,results)

if __name__ == '__main__':main()
