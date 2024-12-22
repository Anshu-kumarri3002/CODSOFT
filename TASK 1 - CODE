# CODSOFT
In this repository i am uploading my all codsoft tasks
**TASK - 1**
# importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# data collection & processing
# load the data from csv file to pandas DataFrame
titanic_data = pd.read_csv('train.csv')
# printing the first 4 rows of the dataframe
titanic_data.head(4)
# number of rows and columns
titanic_data.shape
# getting some information about the data
titanic_data.info()
# check the number of missing values in each column
titanic_data.isnull().sum()
# Handling the missing values
# drop the "cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
# finding the mode value of "Embarked column"
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])
# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
# check the number of missing values in each column
titanic_data.isnull().sum()
#Data Analysis
# getting some statistical measures about the data
titanic_data.describe()
# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()
# data Visualization
sns.set()
# Assuming `titanic_data` is your DataFrame
sns.countplot(x='Survived', data=titanic_data)
# Display the plot
plt.show()
titanic_data['Sex'].value_counts()
# making a countplot for "sex" column
sns.countplot(x='Sex', data=titanic_data)
# Display the plot
plt.show()
# number of survivors Genderwise
sns.countplot(x='Sex', hue = 'Survived', data = titanic_data)
# making a count plot for "Pclass" column
sns.countplot(x= 'Pclass', data = titanic_data)
sns.countplot(x= 'Pclass',hue='Survived', data = titanic_data)
# encoding the Categorical Columns
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
# converting categorical Columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
titanic_data.head()
#seperating features & target
x = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
y = titanic_data['Survived']
print(x)
print(y)
#solitting the data into training data & Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(f"X: {x.shape}, X_train: {x_train.shape}, X_test: {x_test.shape}")
# Logistic Regression
model = LogisticRegression()
# training the Logistic Regression model with training data
model.fit(x_train,y_train)
# accuracy score
x_train_prediction = model.predict(x_train)
print(x_train_prediction)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print('Accuracy score of training data:',training_data_accuracy)
# Accuracy on test data
x_test_prediction = model.predict(x_test)
print(x_test_prediction)
testing_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Accuracy score of test data:',testing_data_accuracy)



