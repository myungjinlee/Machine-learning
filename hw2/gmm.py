## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
import math
#import random

def E(x,mean,var,pi):
    r = list()
    for z in range(0,150):
        a = [0]*3
        r.append(a)
                   
    l = len(x)
    for j in range(0,3):
        
        for i in range(0,l):
            #print(var)
            #tempr = pi[j]*Normal(x[i],mean[j],var[j])
            tempr = multivariate_normal(mean[j],var[j])
            tempr = pi[j]*tempr.pdf(x[i])
            #print("tempr",tempr)
            denom =0
            for k in range(0,3):
                #print(var)
                dtemp = multivariate_normal(mean[k],var[k])
                dtemp = pi[j]*dtemp.pdf(x[i])
                denom = denom + dtemp
            tempr = tempr/denom
            r[i][j] = tempr
    r = np.array(r)        
    return r

def M(x,r):

    pi = [0,0,0]
    mean =[0,0,0]
    var = [0,0,0]
    for c in range(0,3):
        s=0
        s1=0
        s3=0
        #tempvar
        for i in range (0,150):
            
            s = s+r[i][c]
            s1 = (s1 + (r[i][c])*(x[i]))   # use numpy as x is vector
        pi[c] = s/150
        mean[c] = s1/s

        for i in range(0,150):
            stemp2 = np.subtract(x[i],mean[c])
            #print(stemp2)
            stemp2 = np.reshape(stemp2,(1,2))
            stemp1 = np.transpose(stemp2)
            s2 = np.multiply(r[i][c], stemp1)
            s2 = np.multiply(s2,stemp2)
            s3 = s3 + s2
        var[c] = s3/s
        
        mean = np.array(mean)
        var = np.array(var)
        pi = np.array(pi)
        #print(mean)
    return mean,var,pi
        
def Normal(x,mean,var):
    print("bgbhth")
    print(var)
    det = np.linalg.det(var)
    det = 1/math.sqrt(det)
    temp1 = 1/(2*math.pi)
    temp2 = np.subtract(x,mean)
    temp3 = np.transpose(temp2)
    temp4 = np.linalg.inv(var)
    temp5 = np.multiply(temp3,temp4)
    temp5 = (np.multiply(temp5,temp2))/2
    temp5 = 1/math.exp(temp5)

    return temp5*det*temp1

#
fr = open("clusters.txt", "r")
inp = fr.read().splitlines()
x=list()

for i in inp:

    i = i.split(",")
    x.append(i)
    
x= np.array(x)
x = x.astype(np.float)
cl = 3

##main
arr = []
for i in range(0,150):
    a = [0]*3
    arr.append(a)
    
    
for c in range(0,3):
    for i in range(0,150):
        arr[i][c]= np.random.random_integers(1,100)

for i in range(0,150):
    s=0
    for c in range(0,3):
        s=s+arr[i][c]
    for c in range(0,3):
        arr[i][c] = arr[i][c]/s
arr = np.array(arr)        

arrnew = None
flag=0
iteration = 0
while(not(np.array_equal(arr,arrnew))):
    if( flag!= 0):
        
        arr = arrnew
    mean,covar,pi = M(x,arr)
    flag = 1
    arrnew = E(x,mean, covar, pi)
    iteration+=1
    if (iteration==300):
        break
        
    

print("\n Mean for 3 gaussians are \n",mean)
print(" \n Covariance matrices are \n",covar)
print("\n Amplitudes are \n",pi)
    

