#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee

import numpy 
from sklearn import svm
import matplotlib.pyplot as plt

inp_file='linsep.txt'
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
clf=svm.SVC(kernel='linear',C=1000) #c=float
clf.fit(X,y)

print('weight=', clf.coef_)
print('supoort vector=','\n',clf.support_vectors_)
print('b=', clf.intercept_)
print('accuracy=', clf.score(X,y)*100, '%')

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

ax=plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = numpy.linspace(xlim[0], xlim[1], 30)
yy = numpy.linspace(ylim[0], ylim[1], 30)
YY, XX = numpy.meshgrid(yy, xx)
xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z,  colors='red', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()
