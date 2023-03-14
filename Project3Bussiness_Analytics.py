#!/usr/bin/env python
# coding: utf-8

# In[33]:


#import library
import pandas as pd
import numpy as np
import seaborn


# In[34]:


attrition = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/EmployeeAttrition.csv')


# In[35]:


attrition.head()


# In[36]:


attrition.info()


# In[37]:


attrition.describe()


# In[38]:


attrition.isnull().sum()


# In[48]:


var =attrition[['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department']]


# In[49]:


var


# In[ ]:





# In[39]:


attrition.nunique()


# In[ ]:





# In[14]:


# columns name
attrition.columns


# In[15]:


# column names of numerical columns
attrition.select_dtypes(include=np.number).columns


# In[16]:


y =attrition['Attrition']


# In[17]:


X = attrition.select_dtypes(include=np.number)


# In[20]:


# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.5, random_state=25)


# In[21]:


#verify shape
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[22]:


# select model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()


# In[23]:


model


# In[24]:


model.fit(X_train,y_train)


# In[25]:


# predict with model
y_pred=model.predict(X_test)


# In[26]:


# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[27]:


accuracy_score(y_pred,y_test)


# In[28]:


confusion_matrix(y_pred,y_test)


# In[29]:


print(classification_report(y_pred,y_test))


# In[30]:


#future prediction
attrition.sample()


# In[31]:


#define X_new
x_new = X.sample()
x_new.sample()


# In[32]:


# predict for X_new
model.predict(x_new)


# In[ ]:




