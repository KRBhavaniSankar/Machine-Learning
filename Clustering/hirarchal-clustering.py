#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:06:08 2018

@author: bhavani
"""

#Hierachical Clustering
#%reset -f

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing mall data set with pandas
dataset = pd.read_csv("Mall_Customers.csv")

X= dataset.iloc[:,[3,4]]

#Using the dendogram to find optimal number of clusters.

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))  #ward is tryting to minimize WCCS variance in each cluster.

plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Eclidean distance")
plt.show()


#Fitting the Hierarichal clustering to mall dataaset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5,affinity= 'euclidean', linkage= 'ward')

y_hc = hc.fit_predict(X)

#Visualizing the Clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label="Careful")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="blue",label="Standard")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c="green",label="Target")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c="cyan",label="Careless")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c="magenta",label="Sensible")

plt.title("Cluster of Clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
