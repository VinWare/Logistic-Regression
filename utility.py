

import numpy as np
import pandas as pd #not of your use
import logging
import json

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 1e-3
EPOCHS = 100000
MODEL_FILE = 'models/model2'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['company_rating', 'model_rating', 'bought_at', 'months_used', 'issues_rating', 'resale_value']
    X_df = df.get(keys)
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
#    model['output_scaling_factors'] = [y_df.mean(), y_df.std()]
    X = np.array((X_df-X_df.mean())/X_df.std())
#    y = np.array((y_df - y_df.mean())/y_df.std())
    return X, y_df, model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]
#    meany = model['output_scaling_factors'][0]
#    stdy = model['output_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX
#    y = 1.0*(y_df - meany)/stdy

    return X, y_df


def accuracy(X, y, model):

    y_predicted = predict(X,np.array(model['theta']))
    y_actual = np.around(y_predicted + 1) - 1
    error_val = np.square(y_actual - y)
    error_pc = np.sum(error_val) / (1.0 * len(X))
    acc = (1.0 - error_pc) * 100

    print "Accuracy associated with this model is "+str(acc)
    
#    for p in y_actual:
#    	print p
#    print y_predicted
#    print y_actual

def predict(X,theta):
    #pass
    z = z_value(X,theta)
    h = (1 / (1 + np.exp(-z)))
    return h

def z_value(X,theta):
    return(np.dot(X,theta))
