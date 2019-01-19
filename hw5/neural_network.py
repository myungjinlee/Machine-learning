## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.special import expit

data_i = 'downgesture_train.list'
data_o = 'downgesture_test.list'
lr = open(data_i,"r")
lr2 = open(data_o,"r")

files_i = lr.read().splitlines()
files_o = lr2.read().splitlines()
pixels =[]
Y=list()
for f in files_i:
    fil = cv2.imread(f,0)
    fil.reshape(1,960)
    pixels.append(fil)
    if 'down' in f:
        Y.append(1)
    else:
        Y.append(0)
ni=len(files_i)
Y_o=list()
pixels = np.array(pixels)   
pixels = (pixels.T)/255
pixels_o =[]
for f in files_o:
    fil2 = cv2.imread(f,0)
    fil2.reshape(1,960)
    pixels_o.append(fil2)
    if 'down' in f:
        Y_o.append(1)
    else:
        Y_o.append(0)
    
no=len(files_o)
pixels_o = np.array(pixels_o)
pixels_o = (pixels_o.T)/255


def sigmoid(z):
    
    s = expit(z)  
    return s

def initialize_parameters(n_x, n_h, n_y):

    W1=np.zeros((n_h,n_x))
    for i in range(n_h):
        for j in range(n_x):
            W1[i][j]=random.uniform(-0.01,0.01) 
    b1 = np.ones((n_h,1))
    W2 = np.zeros((n_y,n_h))
    for i in range(n_y):
        for j in range(n_h):
            W2[i][j]=random.uniform(-0.01,0.01)
    
    b2 = np.ones((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) 
    
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(W2,A1) 
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dW2 = 2*(A2-Y)*(1- A2)*(A2)  #backward propagation for final layer
    
    db2 = dW2
    
    dW1= np.multiply(np.multiply(1-(A1),A1),(W2*dW2).T)      #check if a1 and w2 have same dimensions

    db1 = dW1
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    #print (grads)
    
    return grads

def update_parameters(X,parameters, grads,cache, learning_rate = 0):
   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    A1 = cache["A1"]

    x=X
    x=np.tile(x,100)
    W1 = W1 - (learning_rate*x.T*dW1)

    b1 = b1 - learning_rate*db1
    W2 = W2 - (learning_rate*A1*dW2).T
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    #print(parameters)
    
    return parameters


def nn_model(Xi, Y, n_h, ni , num_iterations = 1000, print_cost=False):
    
    np.random.seed(3)
    n_x = 960
    n_y = 1
    
    parameters = initialize_parameters(n_x, n_h, n_y)
   
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(1000):
            #print(i)
            test=list()
            for j in range(184):
              
                
                X=Xi[:,:,j].reshape(960,1)
                Ye=Y[j]
                  
                A2, cache = forward_propagation(X,parameters)
            
                grads = backward_propagation(parameters, cache, X, Ye)
         
                parameters = update_parameters(X,parameters, grads,cache, learning_rate =0.1)
                
                test.append(A2)
                
    return test,parameters

def predict(parameters, Xi,no):
    predictions = list()
    for i in range(0,no):
        X=Xi[:,:,i].reshape(960,1)
        A2, cache = forward_propagation(X,parameters)        
        if(A2>0.5):
            predictions.append(1)
        else:
           predictions.append(0)      
    
    return predictions

def accuracy(predictions,actual):
    l=len(predictions)
    c=0.0
    w=0.0
    for i in range(l):
        if(predictions[i]==actual[i]):
            c+=1
        else:
            w+=1
    print('Accuracy',float(c/l)*100)

        
acc,parameters = nn_model(pixels, Y, 100,ni, num_iterations = 1000, print_cost=False)

predictions=predict(parameters,pixels_o,no)
print('Predictions',predictions)
accuracy(predictions,Y_o)

print('Training accuracy')
predictions_ip=predict(parameters,pixels,ni)
accuracy(predictions_ip,Y)

