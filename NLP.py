# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:52:52 2019

@author: jaisw
"""

#NLP
#Review Analysis
#Text Analysis
#We will work with tsv file , rather than csv file,
#because csv file will mix revies & ratings while tsv file seperates the two

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# delimiter='\t : Since we are using read_csv
#quoting=3 :to ignore double quotes
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Our goal is to create bag of words which contains only essentials words(remove punctuation,
#pronouns,conjunctions,taking only the root word,also remove upper casing)

#First we will clean one review & then we apply a loop to clean all reviews

import re

#First review
dataset['Review'][0]

#Example of sub() method
#import re
#text = "Python for beginner is a very cool website"
#pattern = re.sub("cool", "good", text)
#print (text)

#Remove punctuation
review=re.sub('[^a-zA-Z]',' ',dataset['Review'][0])          # ^ implies things we don't want to remove
#' ' to enable spacing between 2 words

#Lower Casing
review=review.lower()


#Removing insignificant words(the,is,at,or,and,them,for......)
import nltk
nltk.download('stopwords')  #contains all insiginficant words that we want to remove

#We have to go through our review to find stopwords to remove them, so we loop
#Our review is a string, so we need to convert to list to llop through it.

review=review.split()  #Converts string into list of words

#Removing garbage words
from nltk.corpus import stopwords
review=[word for word in review if not word in stopwords.words('english')]

#Stemming(keeping only root word, loved-> love)
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=[ps.stem(word) for word in review if not word in stopwords.words('english')]

#Final step is to convert list of words into string
review=' '.join(review)



#Doing the above process for all reviews
import nltk
import re
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])         
    review=review.lower()
    review=review.split()  
    review=[word for word in review if not word in stopwords.words('english')]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


#Create a bag of words model(Takes all the uniqie words & create a column for each word,
#the columns are words and each row is a review,which creates a matrix, since it has many 0's,
# it is called sparsity matrix)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #takes 1500 most frequent words
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
    

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize='False')*100)





