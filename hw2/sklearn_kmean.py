import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

fr = open("clusters.txt", "r")
inp = fr.read().splitlines()
x=list()

for i in inp:

    i = i.split(",")
    x.append(i)
    
x= np.array(x)
x = x.astype(np.float)
cl = 3

kmeans = KMeans(3,random_state=0)
labels = kmeans.fit(x).predict(x)
plt.scatter(x[:,0],x[:,1], c=labels, s=40, cmap = 'viridis')

def plot_kmeans(kmeans,x,n_clusters=3, rseed=0, ax=None):
    labels = kmeans.fit(x).predict(x)

    # plot input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(x[:,0],x[:,1], c=labels, s=40, cmap = 'viridis')#, zorder=2)
    centers=kmeans.cluster_centers_
    ax.scatter(centers[:,0], centers[:,1], marker='*', c='r', s=100)
    
    print(centers)

kmeans = KMeans(n_clusters=3, random_state=None)
plot_kmeans(kmeans,x)
plt.show()
