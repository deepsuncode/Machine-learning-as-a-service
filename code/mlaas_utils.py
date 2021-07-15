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
from __future__ import division
import warnings 
warnings.filterwarnings('ignore')

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import sys
import time
from contextlib import contextmanager
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.linear_model import LogisticRegression

import pickle

from os import listdir
from os.path import isfile, join
import os
import datetime
from  pathlib import Path

custom_models_dir = "custom_models"
custom_models_data_dir = "custom_models_data"
custom_models_time_limit = 24 * 60 #24 hours in minutes
default_models_dir = "models"

algorithms = ['ENS','RF','MLP','ELM']
algorithms_names = ['Ensemble','Random Forest','Multiple Layer Perceptron (MLP)' ,'Extreme Learning Machine (ELM)']

DEFAULT_INPUT_FILE = 'train_data/flaringar_simple.csv'   
logFileHandler = None
timestr = time.strftime("%Y%m%d_%H%M%S")
loggingString = []
algorithm = 'rf,mlp,elm'

flares_col_name ='Flare Class'
logFile = "logs/ens_deepsun.log"
mapping ={1:"B", 2:"C", 3:"M", 4:'X', -1:'N/A'}
B = mapping[1]
C = mapping[2]
M = mapping[3]
X = mapping[4]

class_to_num = {"B":1, "C":2, "M":3, 'X':4, 'N/A':-1}
req_columns =[flares_col_name, "TOTUSJH","TOTBSQ","TOTPOT","TOTUSJZ","ABSNJZH","SAVNCPP","USFLUX","AREA_ACR","TOTFZ","MEANPOT","R_VALUE","EPSZ","SHRGT45"]
no_ver_o = {}
no_ver_o['fcnumber'] = 0
no_ver_o['fcname'] = 'A'
predicted = []
actual = []
confusion_matrix_result= []
cv_mean_value = None
overall_test_accuracy=None
feature_importances = None
partial_ens_trained = False
noLogging = False 
log_to_terminal = False
verbose = False
save_stdout = sys.stdout

@contextmanager
def stdout_redirected(new_stdout):
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout

@contextmanager
def stdout_default():
    sys.stdout = save_stdout

def log(*message,verbose=True, logToTerminal=False, no_time=False, end=' '):
    global noLogging
    if (noLogging) :
        return
    global log_to_terminal
    if log_to_terminal or logToTerminal:
        if not no_time:
            print ('[' + str(datetime.datetime.now().replace(microsecond=0))  +'] ', end=end)
        for msg in message:
            print (msg,end=end)  
        print('')        
    with open(logFile,"a+") as logFileHandler :
        with stdout_redirected(logFileHandler) :
            if no_time:
                print ('[' + str(datetime.datetime.now().replace(microsecond=0))  +'] ',end=end)
            for msg in message:
                print (msg,end=end)  
                global loggingString
                if verbose :
                    loggingString.append( msg) 
            print('')
             
def set_log_to_terminal(v):
    global log_to_terminal
    log_to_terminal = v

def set_verbose(v):
    verbose = boolean(v)
def boolean(b):
    if b == None:
        return False 
    b = str(b).strip().lower()
    if b in ['y','yes','ye','1','t','tr','tru','true']:
        return True 
    return False

def create_default_model(trained_model, model_id):
    return create_model(trained_model, model_id, default_models_dir)

def create_custom_model(trained_model, model_id):
    return create_model(trained_model, model_id, custom_models_dir)

def create_model(trained_model, model_id, model_dir):
    model_file = model_dir + "/" + model_id + ".sav"
    log("create_model saving model with dill "  + model_id + " to file: " + model_file)
    pickle.dump(trained_model, open(model_file, 'wb'))
    return model_file

def is_model_file_exists(file):
    path = Path(custom_models_dir + "/" + file)
    return path.exists()

def is_file_exists(file):
    path = Path(file)
    log("Check if file exists: " + file + " : " + str(path.exists()))
    return path.exists()

def are_model_files_exist(models_dir, modelId, alg='ENS'):
    alg = str(alg).strip().upper()
    log("Searching for model is: " + modelId + " in directory: " + models_dir)
    modelExenstion = ".sav"
    fname = models_dir + "/" + modelId + "_rf" + modelExenstion
    rf_model_exist  = is_file_exists(fname)

    fname = models_dir + "/" + modelId + "_mlp" + modelExenstion
    mlp_model_exist  = is_file_exists(fname) 
    
    fname = models_dir + "/" + modelId + "_elm" + modelExenstion
    elm_model_exist  = is_file_exists(fname)
    
    if alg == 'ENS':
        exist = (rf_model_exist and mlp_model_exist and elm_model_exist)
        if exist:
            return True
        msg ='exist for this model id: ' + modelId + '\nThe ENS algorithm requires the three models: RF, MLP, and ELM to be trained.'
        msg = msg +'\nPlease use the -a open to specify the algorithm you want to test with.\n'
        msg = msg +'Available models for this model id:'
        available_modes  =[]
        models = []
        if not rf_model_exist:
            models.append('RF')
        else:
            available_modes.append('RF')
        if not mlp_model_exist:
            models.append('MLP')
        else:
            available_modes.append('MLP')
        if not elm_model_exist:
            models.append('ELM')
        else:
            available_modes.append('ELM')
            
        if len(available_modes) == 0:
            return False
        global partial_ens_trained
        partial_ens_trained = True
        models_exist = 'model does not' 
        if len(models) > 1:
            models_exist = 'model(s) do not'
            
        print('\n' +  ', '.join(models),models_exist, msg, ', '.join(available_modes))          
        return False
    if alg == 'RF':
        return rf_model_exist
    if alg == 'MLP':
        return mlp_model_exist
    if alg == 'ELM':
        return elm_model_exist
    
    return True  

def get_partial_ens_trained():
    global partial_ens_trained
    return partial_ens_trained
def convert_class_to_num(c):
    c = c[0].strip().upper()
    if c in class_to_num.keys():
        return class_to_num[c]
    return -1
def load_model(model_dir, model_id):
    model_file = model_dir + "/" + model_id + ".sav"
    log("Loading model file: " + model_file)
    if is_file_exists(model_file) :
        model = pickle.load(open(model_file, 'rb'))

        log("Loaded model " + model_file)
        log("Returning loaded model")
        return model  
    log("returning NO MODEL FILE exist")
    return "NO MODEL FILE"  

def load_dataset_csv(data_file):
    log("Reading data set from file: " + data_file)
    dataset = pd.read_csv(data_file)
    return dataset

def load_dataset_csv_default():
    return load_dataset_csv(DEFAULT_INPUT_FILE)

def removeDataColumn (col, data):
    if col in data.columns:
        data = data.drop(col, axis = 1)
    return data


def remove_default_columns(dataset):
    log('Removing default columns from data set')
    dataset = removeDataColumn('goes', dataset)
    dataset = removeDataColumn('fdate', dataset)
    dataset = removeDataColumn('goesstime', dataset)
    dataset = removeDataColumn('flarec', dataset)
    dataset = removeDataColumn('noaaar', dataset)  
    return dataset

def remove_additional_columns(dataset):
    log('Removing default columns from data set')
    remove_default_columns(dataset)
    cols = dataset.columns
    for c in cols:
        if c not in req_columns:
            dataset = removeDataColumn(c, dataset)
    return dataset

def split_data(dataset, target_column = 'flarecn', test_percent=0.1):
    labels = np.array(dataset[target_column])
    dataset = removeDataColumn(target_column, dataset)
    columns = dataset.columns
    train_x, test_x, train_y, test_y = train_test_split(dataset[columns], labels, test_size = test_percent)
    return (train_x, test_x, train_y, test_y)


def normalize_scale_data(d):
    min = np.array(d).min() 
    max = np.array(d).max() 
    d = (d - min) / (max  - min)
    return d

def load_train_test_datasets_csv(testFile, trainFile = DEFAULT_INPUT_FILE, target_column = 'flarecn', additional_col=''):
    dataset = load_dataset_csv(trainFile)
    dataset = remove_default_columns(dataset)
    dataset = removeDataColumn(additional_col, dataset)
    
    testData = pd.read_csv(testFile)
    testData = remove_default_columns(testData)  
    testData = removeDataColumn(additional_col, testData)
    
    labels = np.array(dataset[target_column])
    labels1 = np.array(testData[target_column])

    dataset = removeDataColumn(target_column,dataset)
    testData = removeDataColumn(target_column,testData)
    
    log ("training labels are as follows:")
    log(labels)

    train_x = dataset[dataset.columns]
    train_y = labels
    
    test_x = testData[testData.columns]
    test_y = labels1
    log('test labels are as follows')
    log(labels1)
    return (train_x, test_x, train_y, test_y)

def get_train_test_datasets(trainData, testData, target_column = 'flarecn', additional_col=''):
    trainData = remove_default_columns(trainData)
    trainData = removeDataColumn(additional_col, trainData)
    
    testData = remove_default_columns(testData)  
    testData = removeDataColumn(additional_col, testData)
    
    labels = np.array(trainData[target_column])
    labels1 = np.array(testData[target_column])

    trainData = removeDataColumn(target_column,trainData)
    testData = removeDataColumn(target_column,testData)
    
    log ("training labels are as follows:")
    log(labels)

    train_x = trainData[trainData.columns]
    train_y = labels
    
    test_x = testData[testData.columns]
    test_y = labels1
    log('test labels are as follows')
    log(labels1)
    return (train_x, test_x, train_y, test_y)


def set_print_results(test_y,  predictions): 
    return set_results(test_y,  predictions)
           
def set_results(test_y, predictions, logging=True):
    c = 0
    results = []
    index = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    total = 0
    for i in range(0, len(test_y)):
        if list(test_y)[i] == 1 :
            c1 = c1 + 1
        if list(test_y)[i] == 2 :
            c2 = c2 + 1
        if list(test_y)[i] == 3 :
            c3 = c3 + 1
        if list(test_y)[i] == 4 :
            c4 = c4 + 1        
        e = ""
        if list(test_y)[i] == predictions[i]  :
            e = "match"
            c = c + 1
        if logging: 
            log (str(i) + ") - Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]) + " " + e)
        obj = {}
#         obj['dataitem'] = index
        index = index +1
        obj["fcnumber"] = predictions[i]
        obj["fcname"] = "" + mapping[predictions[i]]
        results.append(obj)
        total = total + 1
    if logging:
        log ("c: " + str(c) +  " total test " + str( len(test_y))) 
        log ( "c1: " +  str(c1) +  ",  c2: " +  str(c2)  +  ", c3: " +  str(c3) +  ", c4: " + str(c4) +  ",  total: " +  str(total))
        log ("Test Accuracy  :: " + str( accuracy_score(test_y, predictions)))
    global overall_test_accuracy
    overall_test_accuracy = accuracy_score(test_y, predictions) 
    global predicted 
    predicted = predictions
    global actual
    actual = test_y
    return results


def print_confusion_matrix(test_y, predictions):
    log (" Confusion matrix ")
    conf_matrix = confusion_matrix(test_y, predictions)
    log(conf_matrix)
    cmp = pd.crosstab(test_y, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
    log("confusion matrix printed")
    log(cmp)
    row =0
    col =0
    global confusion_matrix_result
    confusion_matrix_result = []
    for c in conf_matrix:
        st = ''
        a = []
        for c1 in c:
            a.append(int(c1))
            col = col + 1
            if st == '':
                st = str(c1)
            else:
                st = str(st) + '\t' +  str(c1)
        log (st)
        confusion_matrix_result.append(a)
    return conf_matrix

def rf_train_model(train_x=None, 
               test_x=None, 
               train_y=None, 
               test_y=None, 
               model_id="default_model"):

    alg_model = RandomForestClassifier(n_estimators = 1000, max_features=6, n_jobs=1)
    

    result  = model_train_wrapper('RF', alg_model, 
                            train_x=train_x, 
                            test_x=test_x, 
                            train_y=train_y, 
                            test_y=test_y,
                            model_id=model_id)    
    
    return result 

def mlp_train_model(train_x=None, 
               test_x=None, 
               train_y=None, 
               test_y=None, 
               model_id="default_model"):
    log('Creating MLP hidden layers with neurons')
    h = []
    for i in range(0,200):
        h.append(150)
    h = tuple(h)
    alg_model = MLPClassifier(hidden_layer_sizes=h, 
                              activation='relu', 
                              solver='lbfgs', 
                              batch_size=200)

    scaler = StandardScaler()
    scaler.fit(train_x)
    StandardScaler()
    train_x = scaler.transform(train_x)
    if test_x is not None:
        test_x = scaler.transform(test_x)
        
    result  = model_train_wrapper('MLP', 
                                  alg_model, 
                                  train_x=train_x, 
                                  test_x=test_x, 
                                  train_y=train_y, 
                                  test_y=test_y,
                                  model_id=model_id)    
    
    return result

def elm_train_model(train_x=None, 
               test_x=None, 
               train_y=None, 
               test_y=None, 
               model_id="default_model"):
    log('Creating ELM hidden layers with neurons')
    log('ELM--> Creating model for training..')
    ml_layer = MLPRandomLayer(n_hidden=200, 
                              activation_func='tanh')
    alg_model = GenELMClassifier(hidden_layer=ml_layer)

    scaler = StandardScaler()
    scaler.fit(train_x)
    StandardScaler()
    train_x = scaler.transform(train_x)
    if test_x is not None:
        test_x = scaler.transform(test_x)   

    result  = model_train_wrapper('ELM', 
                                  alg_model, 
                                  train_x=train_x, 
                                  test_x=test_x, 
                                  train_y=train_y, 
                                  test_y=test_y,
                                  model_id=model_id)    
    
    return result 


def valid_data(x):
    return (x is not None and len(x) > 0)
def model_train_wrapper(model_name,
                        alg_model, 
                        train_x=None, 
                        test_x=None, 
                        train_y=None, 
                        test_y=None, 
                        model_id='default_model'):
    if not valid_data(train_x) or not valid_data(train_y):
        print('Invalid training and testing data')
        sys.exit()
        
    log("===============================", algorithms_names[algorithms.index(model_name.strip().upper())] ,"Logging Stared ==============================")
    log("Execution time started: " + timestr)
    

    model_dir = default_models_dir
    if model_id == 'default_model':
        model_dir = default_models_dir
    else:
        model_dir = custom_models_dir 
        
    log('Using model directory:', model_dir, 'for model id: ', model_id)        
    trained_model = alg_model.fit(train_x, train_y)
    log ("Model trained for model id:", model_id) 
    r = {} 
    model_file = model_dir + "/" + model_id + "_" + model_name.strip().lower() + ".sav"
    create_model(trained_model, model_id + "_" + model_name.strip().lower(), model_dir)
    r['model_' + model_name.strip().lower() + '_location'] = model_file
    

    r['errorMessage'] = ''
    r['success'] = "true"
    r['executionStatus']  = 'Pass'
    log("Finished ok")
    log("final result")
    r['algorithms'] = model_name.strip().upper()
    log("Execution time ended: " + timestr + " and ended: " + time.strftime("%Y%m%d_%H%M%S"))
    log("===============================", algorithms_names[algorithms.index(model_name.strip().upper())]  ,"Logging Finished ==============================")

    return trained_model

def model_prediction_wrapper(model_name, 
                             alg_model, 
                             test_x=None,
                             test_y=None,
                             model_id='default_model'):
    log("===============================", 
        algorithms_names[algorithms.index(model_name.strip().upper())] ,
        " Prediction Logging Stared ==============================")
    log("Execution time started: " + timestr)
    
    model_dir = default_models_dir
    if model_id == 'default_model':
        model_dir = default_models_dir
    else:
        model_dir = custom_models_dir 
        
    log('Using model directory:', model_dir, 'for model id: ', model_id)        
    if alg_model is not None:
        log('Using trained model without loading..')
        trained_model = alg_model 
    else:
        log("loading the pre-trained model")
        trained_model = load_model(model_dir, model_id + "_" + model_name.strip().lower())
        log('Done loading the model..')

    r = {} 
    log(model_name, 'Performing the prediction ')
    if not model_name.strip().upper() == 'RF':
        scaler = StandardScaler()
        scaler.fit(test_x)
        test_x = scaler.transform(test_x)

    predictions = trained_model.predict(test_x)
    log(model_name,'Done the prediction')
    global verbose
    results = 'verbose is not set'
    if verbose:
        results = set_print_results(test_y, predictions)
        log(model_name,'Done printing the result')
        print_confusion_matrix(test_y, predictions)
        r['predictionResult'] = results
    r['errorMessage'] = ''
    r['success'] = "true"
    r['executionStatus']  = 'Pass'
    log("Finished ok")
    if verbose: 
        r['logging'] = loggingString
        r['algorithms'] = model_name.strip().upper()
        print (r)
    log("Execution time ended: " + timestr + " and ended: " + time.strftime("%Y%m%d_%H%M%S"))
    log("===============================", 
        algorithms_names[algorithms.index(model_name.strip().upper())],
        "Prediction Logging Finished ==============================")
  
    return predictions

def compute_ens_result(rf_result, mlp_result, elm_result):
    final_results = []
    for i in range(0, len(rf_result)):
        rf_p = rf_result[i]
        ml_p = mlp_result[i]
        el_p = elm_result[i]
        if rf_p == ml_p and rf_p == el_p :
            final_results.append(mapping[rf_p]) 
        elif rf_p == ml_p or rf_p == el_p :
            final_results.append(mapping[rf_p])
        elif el_p == ml_p :
            final_results.append(mapping[el_p])
        else :
            final_results.append(mapping[rf_p])
    
    return final_results

def map_prediction(prediction):
    result = []
    for r in prediction:
        result.append(mapping[r])
    
    return result

def log_cv_report(y_true,y_pred):
    r = multilabel_confusion_matrix(y_true, y_pred)
    log(r)
    ac = accuracy_score(y_true, y_pred)
    log(ac)
    log('Prediction accuracy:', ac)

def save_result_to_file(alg, result, dataset, flares_names):
    result_file =  'results' + os.sep  + str(alg) +'_result.csv'
    print('Writing result to file:', result_file)
    dataset_ens = dataset[:]
    dataset_ens = dataset_ens.drop('flarecn', axis=1)
    dataset_ens.insert(loc=0, column=flares_col_name, value=flares_names)
    dataset_ens.insert(loc=0, column='Prediction', value=result)
    if 'index' in dataset_ens.columns:
        dataset_ens = dataset_ens.drop('index', axis=1)
    dataset_ens.to_csv(result_file,index=False)

def create_default_dirs():
    for d in ['custom_models', 'models', 'logs', 'test_data', 'train_data', 'results']:
        if not os.path.exists(d) :
            os.mkdir(d)

create_default_dirs()  