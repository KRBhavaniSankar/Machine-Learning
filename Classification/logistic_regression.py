#Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")

#print(dataset)
#print(type(dataset))

X=dataset.iloc[:,[2,3]].values

#print(X)
#print(type(X))

y=dataset.iloc[:,4].values
#print(Y)
#print(type(Y))

#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#print(X_train,X_test,y_train,y_test)
#print(type(X_train),type(X_test),type(y_train),type(y_test))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Fitting the logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predicting the test results

y_pred = classifier.predict(X_test)


#Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

#visualizing the Training set results





