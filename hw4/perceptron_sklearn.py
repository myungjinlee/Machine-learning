import pandas 
import numpy 
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=17)


df=pandas.read_csv('classification.txt', header=None)
X0 = numpy.ones((2000,1),dtype=float)
X=df[df.columns[0:3]]
Y=df[df.columns[3]]
X=numpy.array(X)
T=numpy.append(X0, X, axis=1)
#print(T)
Y=numpy.array(Y)
clf=Perceptron(verbose=0, random_state=None, fit_intercept=False, eta0=0.002)
clf.fit(T,Y)

print('accuracy: ' ,clf.score(T, Y)*100)
print('weights :', clf.coef_)
print('intercept : ', clf.intercept_ )


