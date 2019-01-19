## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import pandas 
import numpy 
import random
import copy
numpy.set_printoptions(precision=17)
dataframe=pandas.read_csv('classification.txt', header=None)
x=dataframe.values
xtemp2 = x[:,0:3]
xtemp1 = numpy.ones((2000,1),dtype=float)
xtemp = numpy.append(xtemp1,xtemp2,axis=1)
comp = x[:,3]

w=[]
numpy.random.seed(0)   
for i in range(4):
    w.append(numpy.random.rand())

alpha=0.01
d = xtemp.shape[0]
hyplane=0
w=numpy.array(w)
w.reshape(1,4)
print(w)
flag=False
mis=0

while(flag==False):
    
    
    satisfied=0
    #print('---------------------')
    for i in range (0, 2000):
        
        hyp = numpy.dot(w,xtemp[i])
        
        if((hyp < 0 and comp[i] > 0) or (hyp > 0 and comp[i] < 0)):
            
            if(hyp < 0):
                
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
            flag=True

print('Weights ater final iteration: ', w)
v= numpy.dot(xtemp,w)
v[v>0]=1
v[v<0]=-1
acc = numpy.count_nonzero(v-comp)

print("Accuracy after final iteration",100 - acc*100/2000, "%")
