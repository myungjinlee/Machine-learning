import pandas as pd
import numpy as np
import random

np.set_printoptions(precision=17)
dat=pd.read_csv('classification.txt', header=None)
allval=dat.values
n = allval.shape[0]

features = allval[:,0:3]
target = allval[:,4]

from sklearn import linear_model

logistic = linear_model.LogisticRegression(penalty=None, fit_intercept=True, solver='saga', max_iter=7000)
print("Accuracy",logistic.fit(features,target).score(features,target)*100,"%")
print("Weights are",logistic.coef_, "Bias (W0) is",logistic.intercept_ )
