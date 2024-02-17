# YBI-FOUNDATION-ML-FINAL-PROJECT
---
Dataset Description :- The dataset consists of 1030 instances with 9 attributes and has no missing values. There are 8 input variables and 1 output variable. Seven input variables represent the amount of raw material (measured in kg/m³) and one represents Age (in Days). The target variable is Concrete Compressive Strength measured in (MPa — Mega Pascal). We shall explore the data to see how input features are affecting compressive strength.

Exploratory Data Analysis :- The first step in a Data Science project is to understand the data and gain insights from the data before doing any modelling. This includes checking for any missing values, plotting the features with respect to the target variable, observing the distributions of all the features and so on. Let us import the data and start analysing.

CODE :-

## import library
---
import pandas as pd
## view data
---
cement.head()
## info of data
---
cement.info()
## summary statistics
---
cement.describe()
## check for missing values
---
df = cement df.isnull().sum()
## check for categories
---
cement.nunique()

import seaborn as sns import matplotlib.pyplot as plt
## visualize pairplot
---
sns.pairplot(df) plt.show()
## columns name
---
cement.columns
## define y
---
y = cement['Concrete Compressive Strength(MPa, megapascals) ']
## define X
---
X = cement.drop(['Concrete Compressive Strength(MPa, megapascals) '], axis=1)
## split data
---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2529)
## verify shape
---
X_train.shape, X_test.shape, y_train.shape, y_test.shape
## select model
---
from sklearn.linear_model import LinearRegression model = LinearRegression()
## train model
---
model.fit(X_train,y_train)
## predict with model
---
y_pred=model.predict(X_test)
## model evaluation
---
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

## model MAE
---
mean_absolute_error(y_test,y_pred)
## model MAPE
---
mean_absolute_percentage_error(y_test,y_pred)
## model MSE
---
mean_squared_error(y_test,y_pred)
## define X_new
---
X_new = X.sample()
## future prediction
---
X_new
## predict for X_new
---
model.predict(X_new)


