# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:20:12 2019

@author: jaisw
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Mall_Customers.csv")

dataset.head()

#Matrix of features(will use only age & salary)
X=dataset.iloc[:,3:5].values

#using elbow method to find optimal number of customers

from sklearn.cluster import KMeans

WCSS=[]

for i in range(1,11):
    #max_iter: Maximum iterations of algorithm
    #n_init:no of times algorithm will run with different initial centroids
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    #inertia_ method calcualtes WCSS
    WCSS.append(kmeans.inertia_)

plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
#only for plotting clusters in 2 dimesnions

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'orange', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

