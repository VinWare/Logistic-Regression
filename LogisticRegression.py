import numpy as np
import json
from utility import *

FILE_TRAIN_DATA = 'train.csv'
FILE_TEST_DATA = 'test.csv'
ALPHA = 2e1
EPOCHS - 75000
FILE_MODEL = 'model/model.dat'

train_flag = False ;

def appendIntercept(X):
	return(np.hstack((np.ones((X.shape[0], 1)), X)))

def originalTheta(len):
	return np.zeros(len)

def sigmoid(X):
	return(1.0 / (1 + np.exp(-X)))

def calcGrad(h_theta, X, y):
	factor = h_theta - y
	factor = factor.values.reshape((X.shape[0], 1))
	term = factor * X
	sigma = np.sum(term, axis=0)
	return sigma / X.shape[0]

def train_model():
	model = {}

	X_df, y = loadData(FILE_TRAIN_DATA)
	X, y, model = normalizeData(X_df, y, model)

	X = appendIntercept(X)
	theta = originalTheta(X.shape[1])

	for i in range(EPOCHS):
		h_theta = sigmoid(np.dot(theta, X.T))
		grad = calcGrad(h_theta, X, y)
		theta = theta - ALPHA * grad

	model['theta'] = list(theta)
	with open(FILE_MODEL, 'w') as f:
		f.write(json.dumps(model))

def predict(X, theta):
	z = np.dot(theta, X.T)
	h_theta = sigmoid(z)
	y_predicted = np.around(h_theta + 1) - 1
	return y_predicted

def test_acc(X, y, given_theta):
	theta = np.array(given_theta)
	y_predicted = predict(X, theta)
	y_diff = np.square(y_predicted - y)
	num_error = np.sum(y_diff)
	err_rat = num_error / (1.0 * X.shape[0])
	acc_pc = (1.0 - err_rat) * 100
	print "The accuracy of the model is", acc_pc

#NOT YET READY
def test_model():
	model = {}
	with open(FILE_MODEL, 'r') as f:
		model = json.loads(f.read())
		X_df, y = loadData(FILE_TEST_DATA)
		X, y = normalizeTestData(X_df, y, model)
		X = appendIntercept(X)
		test_acc(X, y, model['theta'])
#READY
if(train_flag):
	train_model()
else:
	test_model()
