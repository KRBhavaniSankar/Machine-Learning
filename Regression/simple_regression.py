import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/media/bhavani/New Volume1/ISS_Bang/Tutorials/DataScience/ML/ML_course/2.Data Preprocessing/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/__MACOSX/Simple_Linear_Regression/Salary_Data.csv")

#print(dataset)
#print(type(dataset))

X=dataset.iloc[:,:-1].values

#print(X)
#print(type(X))

y=dataset.iloc[:,1].values
#print(y)
#print(type(y))

#splitting the data into training and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#print(X_train,X_test,y_train,y_test)
#print(type(X_train),type(X_test),type(y_train),type(y_test))


#Fitting simple linear regression model to the training set


from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(X_train,y_train)
#Predicting the test set results
y_pred = regressor.predict(X_test)
#print(y_pred)
#print(y_test)

#Visualizing the training set results.

plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("salary vs experience {Training set}")
plt.xlabel("Years of Experience")
plt.ylabel("salary")
plt.show()


#Visualizing the test set results.

plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("salary vs experience {Test set}")
plt.xlabel("Years of Experience")
plt.ylabel("salary")
plt.show()
