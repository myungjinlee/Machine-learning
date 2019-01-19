## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import pandas as pd
import numpy as np
from numpy.linalg import multi_dot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=17)
dat=pd.read_csv('linear-regression.txt', header=None)
allval=dat.values
 
features_no_bias = allval[:,0:2].T
xtemp1 = np.ones((1,3000),dtype=float)
features = np.append(xtemp1,features_no_bias,axis=0)

target = np.array(allval[:,2].T)

n=allval.shape[0]

feat = np.linalg.inv(np.dot(features,features.T))

w_opt = multi_dot([feat,features,target])
print("Optimum weights are ",w_opt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(allval[:,0], allval[:,1],allval[:,2])
x1 = np.linspace(0,1,100)
ax.plot(x1,x1,w_opt[0]+(np.multiply(w_opt[1],x1) + np.multiply(w_opt[2],x1)),'.r-')
plt.show()
