#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import pandas as pd 
import numpy as np


# In[2]:


#import data
cement= pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv')


# In[3]:


#view data
cement.head()


# In[4]:


# info of data
cement.info()


# In[5]:


#summary statistics
cement.describe()


# In[6]:


#check for missing values
cement.isnull().sum()


# In[7]:


#check for categories
cement.nunique()


# In[8]:


# visualize pairplot
import seaborn
seaborn.pairplot(cement)


# In[9]:


cement.columns


# In[11]:


y = cement['Concrete Compressive Strength(MPa, megapascals) ']


# In[12]:


X = cement[['Cement (kg in a m^3 mixture)',
       'Blast Furnace Slag (kg in a m^3 mixture)',
       'Fly Ash (kg in a m^3 mixture)', 'Water (kg in a m^3 mixture)',
       'Superplasticizer (kg in a m^3 mixture)',
       'Coarse Aggregate (kg in a m^3 mixture)',
       'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)']]


# In[13]:


y


# In[14]:


X


# In[15]:


#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=20)


# In[16]:


#verify shape
X_test.shape,X_train.shape, y_test.shape , y_train.shape


# In[17]:


#select model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[18]:


model


# In[19]:


#train model
model.fit(X_train,y_train)


# In[21]:


#predict with model
y_pred=model.predict(X_test)


# In[22]:


#model evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# In[23]:


#model MAE
mean_absolute_error(y_test,y_pred)


# In[24]:


#model MAPE
mean_absolute_percentage_error(y_test,y_pred)


# In[25]:


#model MSE
mean_squared_error(y_test,y_pred)


# In[26]:


#future prediction
cement.sample()


# In[27]:


X.sample()


# In[29]:


#define x_new
x_new=X.sample()
x_new.sample()


# In[30]:


# predict for x_new
model.predict(x_new)


# In[ ]:




