#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries for handling data

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import classifier libraries 

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

print('modules uploaded successfully')


# Data Preprocessing

# In[2]:


df = pd.read_csv('gold_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


# Showing a Sample of Data

df.sample(1).iloc[0]


# In[6]:


df.info()


# In[7]:


df.describe()


# Data Cleaning

# In[8]:


#check the null values

df.isnull().sum()

#No null values 


# In[9]:


#check if there is any duplicated values
df.duplicated().sum()


# Data Visualization 
# 

# In[10]:


plt.figure(figsize=(15, 6))
df['GLD'].plot()
df['SLV'].plot()
plt.xlabel('count of Values(GLD & SLV)')
plt.ylabel('GLD & SLV Prices')
plt.title("GLD & SLV Prices")
plt.legend(['Gold', 'Silver'])
plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(15, 6))
df['EUR/USD'].plot()
plt.xlabel('(EUR/USD) Distrubtion')
plt.legend(['EUR/USD'])
plt.tight_layout()
plt.show()


# In[12]:


sns.distplot(df['GLD'], color='blue', kde=True)


# In[13]:


sns.distplot(df['SLV'], color='blue', kde=True)


# In[14]:


sns.distplot(df['EUR/USD'], color='blue', kde=True)


# In[15]:


sns.distplot(df['SPX'], color='blue', kde=True)


# spliting Data into Feature and Target variable

# In[16]:


x = df.drop(['Date', 'GLD'], axis=1)
y = df['GLD']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Random Forest Regressor

# In[18]:


model = RandomForestRegressor()
model.fit(x_train, y_train)


# In[19]:


# testing prediction

test_prediction = model.predict(x_test)
test_prediction


# In[21]:


# r_squared error

score = metrics.r2_score(y_test, test_prediction)
print(f"The Value of R Squared error : {score}")


# In[22]:


# mean squared error

mse = metrics.mean_squared_error(test_prediction, y_test)
print(f"The Value of Mean Squared Error : {mse}")


# Showing the actual price vs the predicted price 

# In[23]:


y_test = list(y_test)


# In[28]:


plt.figure(figsize=(15, 6))
plt.plot(y_test, color='blue', label='Actual Value')
plt.plot(test_prediction, color='yellow', label='Predicted Value')
plt.title("Actual Price vs Predicted Price")
plt.xlabel('Number of Values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()

