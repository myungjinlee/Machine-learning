import pandas as pd
import numpy as np
import random

np.set_printoptions(precision=17)
dat=pd.read_csv('linear-regression.txt', header=None)
allval=dat.values
 
features = allval[:,0:2]
target = np.array(allval[:,2])


from sklearn import linear_model

linmod = linear_model.LinearRegression(fit_intercept=True)

print("Weights are",linmod.fit(features,target).coef_)
print("Bias is",linmod.intercept_)
      
