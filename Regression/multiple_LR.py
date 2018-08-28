import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/media/bhavani/New Volume1/ISS_Bang/Tutorials/DataScience/ML/ML_course/2.Data Preprocessing/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/__MACOSX/Multiple_Linear_Regression/50_Startups.csv")

#print(dataset)
#print(type(dataset))


X=dataset.iloc[:,:-1].values

#print(X)
#print(type(X))

y=dataset.iloc[:,4].values
#print(y)
#print(type(y))


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X =LabelEncoder()
X[:,3]= labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variables trap
X = X[:,1:]
#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#print(X_train)
#print(X_test)
#print(X_train.shape)
#print(X_test.shape)

#print(X_train,X_test,y_train,y_test)
#print(type(X_train),type(X_test),type(y_train),type(y_test))

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predicting the test set results
y_pred = regressor.predict(X_test)
print(y_pred.shape)
#print(y_test.shape)
"""print("Actual - predicted")
for i in range(0,10):
    print("{}\t{}".format(round(y_test[i],2),round(y_pred[i],2)))
print("end")"""

#Building the optimal model using backward elimination


import statsmodels.formula.api as sm
X= np.append(arr=X,values=np.ones((50,1)).astype(int),axis=1)
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0,1,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
print(regressor_OLS.summary())
