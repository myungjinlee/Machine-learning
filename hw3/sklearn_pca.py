from sklearn.decomposition import PCA
import pandas
import numpy

X=pandas.read_csv('pca-data.txt', header=None, sep='\t')
X=numpy.array(X)
print(X)
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

print(pca.components_[0])
print(pca.components_[1])
