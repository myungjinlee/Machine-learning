#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee

import numpy 
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style

inp_file='nonlinsep.txt'
f_inp_obj=open(inp_file, 'r')

x=[]
y=[]
label=[]
for line in f_inp_obj:
    tmp=line.split(',')
    if len(tmp)==3:
        x.append(float(tmp[0]))
        y.append(float(tmp[1]))
        label.append(float(tmp[2]))

xy=numpy.array(list(zip(x, y)))
label=numpy.array(label)

X=xy
y=label

clf=svm.SVC(kernel='poly', degree=2).fit(X,y)#, degree=1, coef0=1, gamma=1)


print('b=', clf.intercept_)
print('support vector=', clf.support_vectors_)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30)

ax=plt.gca()
xx = numpy.linspace(-25, 25, 80)
yy = numpy.linspace(-25, 25, 80)
YY, XX = numpy.meshgrid(yy, xx)
xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()
