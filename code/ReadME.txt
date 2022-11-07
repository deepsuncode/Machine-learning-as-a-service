This ReadMe explains the requirements and how to run the MLaaS flare prediction package using Python from the command line. 
It does not provide steps to run the API or the Web version. In order to use the API or the Web version, please visit the DeepSun website at: https://nature.njit.edu/spacesoft/DeepSun/ 

Prerequisites:

Python version:
The enclosed MLaaS package was built using Python version 3.8.6 (https://www.python.org/downloads/release/python-386/).
Therefore, in order to run the default out-of-the-box models to make predictions, you must use the exact version of Python. 
Other versions are not tested, but they should work if you have the environment set properly to run the package.

Python Packages:
The following python packages and modules are required to run MLaaS:
pandas
numpy
sklearn
scikit-learn
sklearn_extensions

To install the required packages(recommended), u may use Python package manager pip as follows:
1.	Copy the above packages into a text file, i.e., requirements.txt
2.	Execute the command:
	pip install -r requirements.txt
Note: There is a requirements.txt file already created for you to use, including the packages that should be used with Python 3.8.6 pip package manager. 
The file is located in the root directory of the MLaaS package.

Package Structure
After downloading the zip files from https://web.njit.edu/~wangj/MLaaS/, unzip the files into a directory so that the MLaaS package includes the following folders and files:
 
 ReadMe.txt       - this ReadMe file
 requirements.txt - includes Python required packages
 custom_models    - directory for custom trained models
 default_models   - includes default trained models for recovery
 logs             - includes the logging info
 models           - includes default models created for default_model id
 original_data    - includes the original database created for this work
 test_data        - includes a sample test/prediction csv file
 train_data       - includes a sample training csv file
 results          - will include the prediction result file(s)
 mlaas_test.py    - Python program to test/predict a trained model
 mlaas_train.py   - Python program to train a model
 mlaas_utils.py   - utilities program used by the test and training programs

Running a Training Task:
1.	mlaas_train.py is used to run the training. 
Type: python mlaas_train.py -h will show you the available options as follows:
usage: mlaas_train.py [-h] [-t TRAIN_DATA_FILE] [-l LOGFILE] [-v VERBOSE] [-a ALGORITHM] [-m MODELID]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_DATA_FILE, --train_data_file TRAIN_DATA_FILE
       full path to a file includes training data to create a model, must be in csv with comma separator.
  -l LOGFILE, --logfile LOGFILE
       full path to a file to write logging information about current execution.
  -v VERBOSE, --verbose VERBOSE
	   True/False value to include logging information in result json object, note that result will contain a lot of information.
  -a ALGORITHM, --algorithm ALGORITHM
        Algorithm to use for training. Available algorithms: ENS, RF, MLP, and ELM.
		ENS the Ensemble algorithm (default), RF Random Forest algorithm,
		MLP Multilayer Perceptron algorithm, ELM Extreme Learning Machine.
  -m MODELID, --modelid MODELID
        model id to save or load it as a file name. This is to identity each trained model.
2.	Examples to run a training:
	python mlaas_train.py	#to run a training job with default parameters.
	python mlaas_train.py -m test_id -t ..\data\train_data\flaringar_training_sample.csv
	To run a training job with model id test_id and the given training data file.

 
Running a Test/Prediction Task:
1.	To run a test/prediction, you may use the existing data sets from the "test_data” directory or provide your own file. 
2.	mlaas_test.py is used to run the test/prediction. 
Type: python mlaas_test.py -h will show you the available options as follows:
	usage: mlaas_test.py [-h] [-t TEST_DATA_FILE] [-l LOGFILE] [-v VERBOSE] [-a ALGORITHM] [-m MODELID]

optional arguments:
  -h, --help show this help message and exit
  -t TEST_DATA_FILE, --test_data_file TEST_DATA_FILE
        full path to a file includes test data to test/predict using a trained model, must be in csv with comma separator.
  -l LOGFILE, --logfile LOGFILE
        full path to a file to write logging information about current execution.
  -v VERBOSE, --verbose VERBOSE
        True/False value to include logging information in result json object, note that result will contain a lot of information
  -a ALGORITHM, --algorithm ALGORITHM
		Algorithm to use for training. Available algorithms: ENS, RF, MLP, and ELM. 
		ENS the Ensemble algorithm is the default, RF Random Forest algorithm, 
		MLP Multilayer Perceptron algorithm, ELM Extreme Learning Machine.
  -m MODELID, --modelid MODELID
        model id to save or load it as a file name. This is to identity each trained model.
3.	Examples to run a test:
	python mlaas_test.py   #to run the default test dataset using default models.
	python mlaas_test.py -m test_id -t \..data\test_data\flaringar_simple_random_40.csv
	To run a test/prediction on the trained model with id  test_id and a test dataset.

You may change the options as you wish to test/predict the desired test data.

