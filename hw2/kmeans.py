## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import copy
import numpy
import random
from matplotlib import pyplot as plt
inp_file='clusters.txt'
f_inp_obj=open(inp_file, 'r')

x=[]
y=[]
for line in f_inp_obj:
    tmp=line.split(',')
    if len(tmp)==2:
        x.append(float(tmp[0]))
        y.append(float(tmp[1]))
data=numpy.array(list(zip(x, y)))
cent_x=[]
cent_y=[]
k=3

for i in range(k):
    cent_x.append(random.randrange(int(numpy.min(x)), int(numpy.max(x))))
    cent_y.append(random.randrange(int(numpy.min(y)), int(numpy.max(y))))
cent=numpy.array(list(zip(cent_x, cent_y)), dtype=numpy.float32)

clus=numpy.zeros(len(data))
labels=numpy.zeros(len(data))
flag=True
while flag:
    for i in range(len(data)):
        dist=numpy.zeros(3).reshape(3,1)
        for j in range(k):
            dist[j]=numpy.linalg.norm(data[i]-cent[j])
            if numpy.isnan(dist).any():
               nan_x=[]
               nan_y=[]
               for l in range(k):
                   nan_x.append(random.randrange(int(numpy.min(x)), int(numpy.max(x))))
                   nan_y.append(random.randrange(int(numpy.min(y)), int(numpy.max(y))))
               cent=numpy.array(list(zip(nan_x, nan_y)), dtype=float) 
        clus[i]=numpy.argmin(dist)
    cent_old=copy.copy(cent)
    for j in range(k):
        tmp_x=[]
        tmp_y=[]
        for i in range(len(data)):
            if clus[i]==j:
                tmp_x.append(data[i][0])
                tmp_y.append(data[i][1])
        cent[j][0]=numpy.mean(tmp_x)
        cent[j][1]=numpy.mean(tmp_y)
        del tmp_x
        del tmp_y
    if (cent==cent_old).all():
       flag=False
print(cent)
for i in range(len(data)):
    distance = numpy.zeros((3,1))
    for j in range(k):
        distance[j] = numpy.linalg.norm(data[i] - cent[j])
    labels[i] = numpy.argmin(distance)        
                                        
plt.scatter(data[:,0], data[:,1],c=labels ,s=50, cmap='viridis')
plt.scatter(cent[:,0], cent[:,1], marker='*', c='r', s=100)
plt.show()
