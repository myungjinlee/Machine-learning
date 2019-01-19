## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import pandas as pd
import numpy as np
import random

np.set_printoptions(precision=17)
dat=pd.read_csv('classification.txt', header=None)
allval=dat.values
n = allval.shape[0]

features_no_bias = allval[:,0:3].T
xtemp1 = np.ones((1,2000),dtype=float)
features = np.append(xtemp1,features_no_bias,axis=0)
target = allval[:,4].T
target =target.reshape(1,allval.shape[0])

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initparam():
    w = np.random.randn(4,1)*0.01
    #w = np.zeros((3,1))
    return w

def log_rig(w,features,target):
    
    ywx = np.multiply(target,np.dot(w.T,features))
    act = sigmoid(ywx)
    cost = -1/n*(np.sum(np.log(act),axis=1,keepdims=True))
    
    dw = 1/n*(np.sum(-np.exp(-ywx)/(1+np.exp(-ywx))*np.multiply(target,features)))

    grads = {"dw":dw}
    return grads

def update(w,features,target,itr,alpha):
    for i in range(itr):
        grads = log_rig(w,features,target)
        dw = grads["dw"]
        w = w - alpha*dw

    params = {"w":w}
    grads = {"dw":dw}

    return params,grads

def model(features,target,itr=7000,alpha=0.05):
    w = initparam()
    paramters,grads = update(w,features,target,itr,alpha)
    w = paramters["w"]

    return w

w = model(features,target,itr=7000,alpha=0.05)
print("Weights are",w)
    
                     
ypred = sigmoid(np.dot(w.T,features))
ycl = ypred
for num in range(0,2000):
    ycl[ycl>0.5]=1
    ycl[ycl<0.5]=-1


bn=ycl-target
vb = np.count_nonzero(ycl-target)

print("Number of correctly classified points is", n-vb)
print("Accuracy is",(n-vb)/n*100 ,"%")



                     
                  
    
                     

