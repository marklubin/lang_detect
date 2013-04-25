"""
	neural.py
	Command Line interface to NeuralNetwork
	
	Mark Lubin

	USAGE:

		1.Train a neural network with layers
			python neural.py -t <TRAINING_DATA_FILE>  <OUTPUT_FILE> <LAYER_SIZES>
			<LAYER_SIZES> in single string seperate by comma 
			eg. 25 hidden layers and 10 output classes
			25,10

		2. Run trained network on data set show accuracy.
			python neural.py -p <TEST_DATA_FILE> <THETA_FILE>

		3. Run tests
			python neural.py -test [TRAIN/PREDICT]
"""
from Detector.NeuralNetwork.NeuralNetwork import *
from Detector.configuration import *
from sys import argv 

def main():
	if len(argv) < 2: 
		print "Invalid Usage."
		return

	switch = argv[1]

	if(switch == '-t'):
		if len(argv) != 5:
			print "Invalid Usage for -t."
			return
		infile = path.join(FEATURES_DIR,argv[2])
		outfile = path.join(CLASSIFIERS_DIR,argv[2])
		try:
			sizes = [int(x) for x in argv[4].split(',')]
		except Exception:
			print "Couldn't parse layer sizes."
			return
		trainer(infile,outfile,sizes)
	elif(switch == '-p'):
		if len(argv) != 4:
			print "Invalid Usage for -p."
		datafile = path.join(FEATURES_DIR,argv[2])
		thetafile = path.join(CLASSIFIERS_DIR,argv[2])
		predictor(datafile,thetafile)
	elif(switch == '-test'):
		if len(argv) != 3:
			print "Invalid Usage to -test."
		if argv[2] == 'TRAIN':
			training_test(TRAINING_TEST_FILE)
		elif argv[2] == 'PREDICT':
			prediction_test(TRAINING_TEST_FILE, WEIGHTS_TEST_FILE)
		else:
			print "No such test %s." % argv[2]
			return
	else:
		print "No such option %s" % switch



if __name__ == "__main__":main()