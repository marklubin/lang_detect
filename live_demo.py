"""
live_demo.py
automatically extract features and return class

Mark Lubin
"""
from feature_extraction.smile_extractor import parse_features
from neural_network.NeuralNetwork import NeuralNetwork as NN
import numpy as np
from os import system,unlink,environ
import subprocess as sp

LANGS = ["ENGLISH","RUSSIAN"]
CONF_BASE = environ['SMILE_CONF'] + "\\"
CONF_RECORD_FILE = "demo\\audiorecorder.conf"
WAVE_FILE = "out.wav"

def main():
	fcall = "SMILExtractPA -C %s -sampleRate 44100 -channels 1 -O %s -noconsoleoutput -l 0"\
		%(CONF_BASE + CONF_RECORD_FILE,WAVE_FILE)
	fcall = fcall.split(' ')
	
	print "\n\nPress any key to start recording...end recording with CTRL-C in new window."
	raw_input()
	rec_proc = sp.Popen(fcall,creationflags=(sp.CREATE_NEW_CONSOLE))

	rec_proc.wait()

	X = parse_features(WAVE_FILE)

	##TODO LOAD NN and predict

if __name__ == '__main__':main()