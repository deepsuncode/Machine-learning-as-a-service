'''
 (c) Copyright 2021
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

import numpy as np
import os
import csv 
from datetime import datetime
import argparse
import time 

from mlaas_utils import *

TRAIN_INPUT = 'train_data/flaringar_training_sample.csv'
TEST_INPUT = 'test_data/flaringar_simple_random_40.csv'
normalize_data = False
def train_model(args):
    algorithm = args['algorithm']
    if not algorithm.strip().upper() in algorithms:
        print('Invalid algorithm:', algorithm, '\nAlgorithm must one of: ', algorithms)
        sys.exit()
    TRAIN_INPUT = args['train_data_file']
    if TRAIN_INPUT.strip() == '':
        print('Training data file can not be empty')
        sys.exit() 
    if not os.path.exists(TRAIN_INPUT):
        print('Training data file does not exist:', TRAIN_INPUT)
        sys.exit()
    if not os.path.isfile(TRAIN_INPUT):
        print('Training data is not a file:', TRAIN_INPUT)
        sys.exit()
    modelid = args['modelid']
    if modelid.strip() == '':
        print('Model id can not be empty')
        sys.exit()
    if modelid.strip().lower() == 'default_model':
        ans = input('Using default_model as an id will overwrite the default models. Are you want to want to continue? [n] ')
        if not boolean(ans):
            print('Existing..')
            sys.exit()
    normalize_data = boolean(args['normalize_data'])
    log('normalize_data:', normalize_data)
    set_log_to_terminal(boolean(args['verbose']))
    if not boolean(args['verbose']):
        print('Verbose is turned off, process is running without verbose, training result will be printed, please wait...')
    log('Your provided arguments as: ', args)
    log("=============================== Logging Stared using algorithm: " + algorithm +" ==============================")
    log("Execution time started: " + timestr)
    log("Log files used in this run: " + logFile)
    log("train data set: " + TRAIN_INPUT)
    log("Creating a model with id: " + modelid)
    print("Starting training with a model with id:",  modelid, 'training data file:', TRAIN_INPUT)
    print('Loading data set...')
    dataset = load_dataset_csv(TRAIN_INPUT)
    log("orig cols: " , dataset.columns)
    for c in dataset.columns:
        if not c in req_columns:
            dataset = removeDataColumn(c,dataset)
    log("after removal cols: " , dataset.columns)
    cols = list(dataset.columns)
    if not flares_col_name in cols:
        print('The required flares class column:', flares_col_name, ' is not included in the data file')
        sys.exit()
    dataset['flarecn'] = [convert_class_to_num(c) for c in dataset[flares_col_name]]
    log('all columns: ', dataset.columns) 
    log('\n', dataset.head())
    dataset = removeDataColumn(flares_col_name, dataset)
    if normalize_data:
        log('Normalizing and scaling the data...')
        for c in cols:
            if not c =='flarecn' and not c== flares_col_name:
                dataset[c] = normalize_scale_data(dataset[c].values)
#     (train_x, test_x, train_y, test_y) = split_data(dataset)
    train_y = dataset['flarecn']
    train_x = removeDataColumn('flarecn',dataset)
    test_x = None 
    test_y = None
    models_dir = custom_models_dir
    printOutput = False
    if modelid == 'default_model' :
        models_dir = default_models_dir
    alg = algorithm.strip().upper()
    if alg in ['RF','ENS']:
        rf_model = rf_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('RF  model train done.')
    
    if alg in ['MLP','ENS']:
        mlp_model = mlp_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('MLP model train done.')

    if alg in ['ELM','ENS']:
        
        elm_model = elm_train_model(train_x, test_x, train_y, test_y,model_id=modelid)
        print('ELM model train done.')
    ens = ''
    if alg == 'ENS':
        ens = '(s)'
    print('Finished training the', alg ,'model' + str(ens)+  ', you may use the mlaas_test.py program to make prediction.')
                    
'''
Command line parameters parser
'''
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_data_file',default=TRAIN_INPUT, help='full path to a file includes training data to create a model, must be in csv with comma separator')
parser.add_argument('-l', '--logfile', default=logFile,  help='full path to a file to write logging information about current execution.')
parser.add_argument('-v', '--verbose', default=False,  help='True/False value to include logging information in result json object, note that result will contain a lot of information')
parser.add_argument('-a',  '--algorithm', default='ENS',  help='Algorithm to use for training. Available algorithms: ENS, RF,MLP, and ELM. \nENS \tthe Ensemble algorithm is the default, RF\t Random Forest algorithm, \nMLP\tMultilayer Perceptron algorithm, \nELM\tExtreme Learning Machine')
parser.add_argument('-m', '--modelid', default='default_model', help='model id to save or load it as a file name. This is to identity each trained model.')
parser.add_argument('-n', '--normalize_data', default=normalize_data, help='Normalize and scale data.')

args = vars(parser.parse_args())

if __name__ == "__main__":
    train_model(args)
