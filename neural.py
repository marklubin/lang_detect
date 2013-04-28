"""
	neural.py
	Command Line interface to NeuralNetwork
	
	Mark Lubin

	USAGE:

		1.Train a neural network with layers
			python neural.py --MODE train -training TRAINING_DATA_FILE -o OUTPUT_FILE [-lmbda LMBDA][-n NUMBER_TRAINING_EXAMPLES] -layers L1 L2 ... LN 

		2. Run trained network on data set show accuracy.
			python neural.py --MODE predict -test TEST_DATA_FILE -classifier CLASSIFIER_FILE
"""
from Detector.NeuralNetwork import test,utils
from Detector.configuration import *
import argparse

def main():

	parser = argparse.ArgumentParser('Neutral Network command Line Interface.')
	parser.add_argument('--MODE',metavar='Usage mode.',type=str,\
						help='one of ' + str(NEURAL_MODES) + " how you want to use the NeuralNetwork.")
	parser.add_argument('--RUNTEST',action='store_true')
	parser.add_argument('-training',metavar='TRAINING_FILE',type=str,\
						help='Location of training data file relative to features folder when using training mode.')
	parser.add_argument('-o',metavar="OUTFILE",type=str,\
		help ='location of output file for classifier when training, relative to classifiers folder when using training mode.')
	parser.add_argument('-test',metavar='TEST_FILE',type=str,\
						help='Location of test data file relative to features folder when using prediction mode.')
	parser.add_argument('-classifier',metavar='WEIGHTS_FILE',type=str,\
						help='Location of weights/theta file to use relative to classifiers folder when using prediction mode.')
	parser.add_argument('-lmbda',metavar='LAMBDA',type=float,default=1.,required=False,\
						help='regularization parameter when using training mode, defaults to 1.')
	parser.add_argument('-logfile',metavar='LOGFILE',type=str,default=CLASSIFIER_LOG_FILE,\
						help="""where to write the results of running the classifier in prediction mode
							  appends file if exists otherwise creates it, path relative to testing directory, 
							  defaults to CLASSIFIER_LOG_FILE""")
	parser.add_argument('-n',metavar='NUMBER_TRAINING_EXAMPLES',type=int,\
						help="""when using training mode, this optional parameter can control the number of training examples to use,
								of course it must be smaller than the number of training examples available in the training file.""")
	parser.add_argument('-resultsfile',metavar="RESULTS_FILE",type=str,\
						help="""file to store actual y and predicted y, for use with prediction mode,
								path relative to testing folder""")
	parser.add_argument('-layers',metavar='LAYERS',type=int,nargs=argparse.REMAINDER,\
						help="""Hidden layer sizes seperated by spaces, eg. for 25 node single hidden layer
								do '-layers 25' for 2 hiddens layers, first of size 25 second of size 10 do
								'-layers 25 10', only needed in training mode, must be final arg!""")
	args = parser.parse_args()

	#just run test cases and exit if this flag is set
	if args.RUNTEST:
		test.prediction_test(TRAINING_TEST_FILE,WEIGHTS_TEST_FILE)
		test.training_test(TRAINING_TEST_FILE)
		return

	mode = args.MODE
	if mode == TRAIN:
		if not args.training or not args.o or not args.layers:
			print "Not enough params for training."
			return
		infile = path.join(FEATURES_DIR,args.training)
		outfile = path.join(CLASSIFIERS_DIR,args.o)
		sizes = args.layers
		lmbda = args.lmbda
		if args.n:
			utils.subset_trainer(infile,outfile,sizes,lmbda,args.n)
		else:
			utils.trainer(infile,outfile,sizes,lmbda)	
			
	elif mode == PREDICT:
		if not args.test or not args.classifier or not args.resultsfile :
			print "Not enough params for prediction."
			return
		datafile = path.join(FEATURES_DIR,args.test)
		thetafile = path.join(CLASSIFIERS_DIR,args.classifier)
		resultsfile = path.join(TESTING_DIR,args.resultsfile)
		logfile = path.join(TESTING_DIR,args.logfile)
		utils.predictor(datafile,thetafile,logfile,resultsfile)



if __name__ == "__main__":main()