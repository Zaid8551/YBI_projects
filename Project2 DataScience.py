#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import library
import pandas as pd
import numpy as np


# In[4]:


# import data
disease = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/MultipleDiseasePrediction.csv')


# In[5]:


disease.head()


# In[6]:


disease.describe()


# In[7]:


disease.info()


# In[8]:


#check for missing values
disease.isnull().sum()


# In[9]:


#check for categories
disease.nunique()


# In[10]:


#correlation
disease.corr()


# In[11]:


#column names
disease.columns


# In[12]:


# define y
y=disease["prognosis"]


# In[13]:


# define X
X = disease.drop(['prognosis'], axis=1)


# In[14]:


# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=20)


# In[15]:


X_test.shape,X_train.shape, y_test.shape , y_train.shape


# In[16]:


# select model
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier() 


# In[17]:


model


# In[18]:


model.fit(X_train,y_train)


# In[19]:


# predict with model
y_pred=model.predict(X_test)


# 

# In[20]:


# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[21]:


accuracy_score(y_test, y_pred)


# In[22]:


confusion_matrix(y_test, y_pred)


# In[23]:


# model classification report
print(classification_report(y_test, y_pred))


# In[24]:


#future prediction
disease.sample()


# In[25]:


# define X_new
x_new = X.sample()
x_new.sample()


# In[26]:


# predict for X_new
model.predict(x_new)


# In[ ]:




