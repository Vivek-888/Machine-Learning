# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:39:20 2019

@author: jaisw
"""

    #USED FOR FINDING OPTIMAL HYPERPRAMATER
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Social_Network_Ads.csv")

dataset.head()

#Matrix of features(will use only age & salary)
X=dataset.iloc[:,2:4].values
#Dependent variable
Y=dataset.iloc[:,4].values


#Splitting into training set & test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Fitting SVM to training set

from sklearn.svm import SVC

#changing kernet type can impact accuracy
classifier=SVC(C=1,kernel='rbf',gamma=0.5,random_state=0) #HYPERPARAMETERES
classifier.fit(x_train,y_train)

#Predicting Test Set Results
y_pred=classifier.predict(x_test)


#Applying K-fold cross validation
from sklearn.model_selection import cross_val_score
#we are doing 10 folds and storing the results of each fold in accuracies
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()

#gives standard deviation
accuracies.std()
#very less variance hence good model



#Applying Grid Search to find best paramaters
from sklearn.model_selection import GridSearchCV
#create a list of dictionaries for parameters
parameters=[ {'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001]},
        ]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search.fit(x_train,y_train)

best_accuracy=grid_search.best_score_

best_parameters=grid_search.best_params_




#Evaluating Performance of our model
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize='False')*100)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))