## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import pandas 
import numpy 
import random
import copy
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=17)

dataframe=pandas.read_csv('classification.txt', header=None)
x=dataframe.values
xtemp2 = x[:,0:3]
xtemp1 = numpy.ones((2000,1),dtype=float)
xtemp = numpy.append(xtemp1,xtemp2,axis=1)
comp = x[:,4]

w=[]  
for i in range(4):
    w.append(numpy.random.rand())

alpha=0.01
d = xtemp.shape[0]
wt = w
iter_cnt =[]
num_misclassified=[]
w=numpy.array(w)
w.reshape(1,4)


mis=xtemp.shape[0]
itr=0
while(itr<7000):
        
    satisfied=0
    hyp=[]
    hyp = numpy.dot(xtemp,w.T)
    hyp[hyp>0]=1
    hyp[hyp<0]=-1
    misclassified = numpy.count_nonzero(hyp-comp)

    if(misclassified < mis):
        mis = misclassified
        wt = w
    
    #print('---------------------')
    num_misclassified.append(misclassified)
    iter_cnt.append(itr)
       
    for i in range (0, 2000):
             
        if((hyp[i] < 0 and comp[i] > 0) or (hyp[i] > 0 and comp[i] < 0)):
            
            if(hyp[i] < 0):
                
                w[0]+=alpha*xtemp[i][0]
                w[1]+=alpha*xtemp[i][1]
                w[2]+=alpha*xtemp[i][2]
                w[3]+=alpha*xtemp[i][3]
            else:
                
                w[0]-=alpha*xtemp[i][0]
                w[1]-=alpha*xtemp[i][1]
                w[2]-=alpha*xtemp[i][2]
                w[3]-=alpha*xtemp[i][3]
            break
        
        satisfied += 1
        if(satisfied == 2000):
            break
    itr+=1
print('Weights after all 7000 iterations', wt)
print('misclassified ',misclassified)
print("Accuracy after 7000 iterations is",(2000-misclassified)/2000*100, "%")
plt.plot(iter_cnt,num_misclassified)
plt.show()
