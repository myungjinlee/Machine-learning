## Group members :  Myungjin Lee, Krishna Akhil Maddali, Yash Shahapurkar

import numpy
import pandas

point=pandas.read_csv('pca-data.txt', header=None, sep='\t')
point=point.T

# compute Mu
mu=numpy.mean(point,axis=1)
d_mat=point.sub(mu,axis=0)

# compute covariance matrix
cov_matrix=numpy.dot(d_mat,d_mat.T)/(point.shape[1])

# compute the eigenvalue and eigenvector 
evalue, evector=numpy.linalg.eig(cov_matrix)

#sorting the eigenvectors by decreasing eigenvalues
pairs = [(numpy.abs(evalue[i]), evector[:,i]) for i in range(len(evalue))]
pairs.sort(key=lambda x: x[0], reverse=True)

matrix_w = numpy.hstack((pairs[0][1].reshape(3,1), pairs[1][1].reshape(3,1)))
print('First two principal components')
print(matrix_w.T[0])
print(matrix_w.T[1])


