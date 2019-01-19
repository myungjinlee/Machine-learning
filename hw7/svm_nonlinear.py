#Group members : Krishna Akhil Maddali , Yash Shahapurkar, Myungjin Lee
import numpy 
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
import math

inp_file='nonlinsep.txt'
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

def z_func(x):
    psi=numpy.zeros(shape=(len(x), 6))
    for i in range(len(x)):
        psi[i][0]=1
        psi[i][1]=x[i][0]**2
        psi[i][2]=x[i][1]**2
        psi[i][3]=numpy.sqrt(2)*(x[i][0])
        psi[i][4]=numpy.sqrt(2)*(x[i][1])
        psi[i][5]=numpy.sqrt(2)*(x[i][0]*x[i][0])
    return psi

def gaus_func(x,xprime,gamma):
    rbf=numpy.zeros(shape=(len(x),1))
    for i in range(len(x)):
        fir=numpy.dot(x[i].T, x[i])
        sec=numpy.dot(xprime[i].T, xprime[i])
        thr=(-2)*numpy.dot(x[i].T, xprime[i])
        tot=fir+sec+thr
        rbf[i]=numpy.exp((-1)*gamma*tot)
    return rbf

def solve_quad(array,l):
    K=numpy.dot(array,array.T)        
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

def get_weight(a,label,array):
    w=0
    for i in range(len(a)):
        w+=a[i]*label[i]*array[i]
    return w

def get_b_alpha(a,label,array,w,x):
    sv_g=[]
    sv_a=[]
    sv_l=[]
    sv_x=[]
    b=[]
    for i in range(len(label)):
        if (a[i]>1e-6): 
            sv_g.append(array[i])
            sv_a.append(a[i])
            sv_l.append(label[i])
    for i in range(len(sv_l)):
        b.append((1/sv_l[i])-numpy.dot(w.T, sv_g[i]))
    return b, sv_a


if __name__ == '__main__':
    data,classifier=read_data(f_inp_obj)
    z=z_func(data)
    all_alphas=solve_quad(z, classifier)
    weight=get_weight(all_alphas, classifier, z)
    b,alpha=get_b_alpha(all_alphas, classifier, z, weight,data)
    
    print('kernel function: polynomial function')    
    print ('weight=', weight)
    #print('alpha=')
    #for p in alpha: print(p)
    print ('b=',b[0])


    fig, ax = plt.subplots()

    r = math.sqrt(-b[0]/0.16)  
    c = plt.Circle((0, 0), r, color='black', fill=False)
    ax.add_artist(c)

    plt.scatter(data[:, 0], data[:, 1], c=classifier.reshape(100), s=30, cmap=plt.cm.Paired)
    ax=plt.gca()
    plt.show()
  



