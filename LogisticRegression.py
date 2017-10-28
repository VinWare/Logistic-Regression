
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 2e1
EPOCHS = 75000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model1'
train_flag = False
calc_error = False

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #pass#remove this line once you finish writing
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    #pass
    return np.zeros(n_thetas)



def train(theta, X, y, model):
     if(calc_error):
	     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     for i in range(EPOCHS):
     	y_predicted = predict(X, theta)
	if(calc_error):
		cost = costFunc(m, y, y_predicted)
		J.append(cost)
	grads = calcGradients(X, y, y_predicted, m)
	theta = makeGradientUpdate(theta, grads)
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)

     if(calc_error):
	     model['J'] = J
     model['theta'] = list(theta)
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    #pass
#    sigma = np.sum((y * np.log(y_predicted)) + ((1 - y) * np.log(1 - y_predicted)))
    log_h = np.log(y_predicted)
    log_comp = np.log(1 - y_predicted)
    term = y * log_h + (1 - y) * log_comp
    sigma = np.sum(term)
    return(sigma / (-m))

def calcGradients(X,y,y_predicted,m):
	#apply the formula , this function will return cost with respect to the gradients
	# basically an numpy array containing n_params
	#pass
	difference = np.subtract(y_predicted, y)
	difference = difference.values.reshape((X.shape[0], 1))
	summation = np.multiply(X,difference)
	return np.sum(summation, axis=0) / m
	

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    #pass
    theta = theta - ALPHA * grads
    return theta


#this function will take two paramets as the input
def predict(X,theta):
    #pass
    z = z_value(X,theta)
    h = (1 / (1 + np.exp(-z)))
    return h

def z_value(X,theta):
    return(np.dot(X,theta))

########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            accuracy(X,y,model)

if __name__ == '__main__':
    main()
