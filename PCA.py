# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:25:37 2019

@author: jaisw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Wine.csv")

dataset.head()

#Matrix of features
X=dataset.iloc[:,0:13].values
#Dependent variable
Y=dataset.iloc[:,13].values


#Splitting into training set & test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#DO this first, then restart kernel and do next
#Applying PCA
#pca=PCA(n_components=None)
#x_train=pca.fit_transform(x_train)
#x_test=pca.transform(x_test)
#explained_variance=pca.explained_variance_ratio_

#take first 2 PCA'S as it explains maximum variance
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
explained_variance=pca.explained_variance_ratio_

#Fitting Logistic Regression to training set

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#Predicting Test Set Results
y_pred=classifier.predict(x_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize='False')*100)

#visualizing the results

