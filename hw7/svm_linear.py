#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee
import numpy 
from cvxopt import matrix,solvers
import matplotlib.pyplot as plt

inp_file='linsep.txt'
f_inp_obj=open(inp_file, 'r')

def read_data(f_inp_obj):
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
    label=numpy.reshape(label, (len(label),1))
    return (xy, label)

def solve_quad(x,l):
    K=numpy.dot(x,x.T)
    P=matrix(K*numpy.dot(l, l.T))
    q=matrix(numpy.ones(len(l))*-1)
    G=matrix(numpy.diag(numpy.ones(len(l))*-1))
    h=matrix(numpy.zeros(len(l)))
    b=matrix(numpy.zeros(1))
    A=matrix(l.T, (1, len(l)))
    solvers.options['show_progress']=False
    sol=solvers.qp(P, q, G, h, A, b)
    alpha=numpy.array(sol['x'])
    return alpha

def get_weight(a,label,x):
    w=0
    for i in range(len(a)):
        w+=a[i]*label[i]*x[i]
    return w

def get_b_alpha(a,label,x,w):
    sv_x=[]
    sv_a=[]
    sv_l=[]
    b=[]
    for i in range(len(label)):
        if (a[i]>1e-6):
            sv_x.append(x[i])
            sv_a.append(a[i])
            sv_l.append(label[i])
    for i in range(len(sv_l)):
        b.append((1/sv_l[i])-numpy.dot(w.T, sv_x[i]))
    return b, sv_a

if __name__ == '__main__':
    data,classifier=read_data(f_inp_obj)
    all_alphas=solve_quad(data, classifier)
    weight=get_weight(all_alphas,classifier,data)
    print ('weight=', weight)
    
    b,alpha=get_b_alpha(all_alphas, classifier, data, weight)
    print ('b=',b[0])

    fig, ax=plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=classifier.reshape(100), s=30, cmap=plt.cm.Paired)

    slope = -weight[0] / weight[1]
    intercept = -b[0] / weight[1]
    x = numpy.linspace(min(data[:,0]), max(data[:,1]), 30)

    ax.plot(x, x * slope + intercept, 'k-')
    plt.ylim((0,1))
    plt.show()
