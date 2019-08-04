# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:47:41 2019

@author: jaisw
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")

#Independent variable-years of exp
X=dataset.iloc[:,:-1].values.astype(int)
#Dependent variable-Salary
Y=dataset.iloc[:,1].values

#Splitting into training set & test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting test set results
y_pred=regressor.predict(x_test)                  

#Visulaising Training Set Results

plt.scatter(x_train,y_train,color='red') # Actual observation points

#Plotting Best Fit result
plt.plot(x_train,regressor.predict(x_train),color='blue')  #Best fit line
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Years of Experience')  
plt.ylabel('Salary')
plt.show()                 
  
from sklearn.metrics import r2_score
r2_score(y_train,regressor.predict(x_train))

#Visulaising Test Set Results

plt.scatter(x_test,y_test,color='red') # Actual observation points

#Plotting Best Fit result
plt.plot(x_train,regressor.predict(x_train),color='blue')  #Best fit line
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Years of Experience')  
plt.ylabel('Salary')
plt.show() 

from sklearn.metrics import r2_score
r2_score(y_test,regressor.predict(x_test))




