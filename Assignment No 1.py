#!/usr/bin/env python
# coding: utf-8

# # Assignment No:- 1

# # House Price Prediction

# # Name:- Rahul Patel
# Roll No:-2100290100126

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data=pd.read_csv(r"C:\Users\smara\OneDrive\Desktop\Housing.csv")
data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.columns


# In[6]:


data.describe()


# In[7]:


data.isna().sum()


# In[ ]:





# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[9]:


data.price=le.fit_transform(data.price)
data.area=le.fit_transform(data.area)
data.bedrooms=le.fit_transform(data.bedrooms)
data.bathrooms=le.fit_transform(data.bathrooms)
data.stories=le.fit_transform(data.stories)
data.mainroad=le.fit_transform(data.mainroad)
data.guestroom=le.fit_transform(data.guestroom)
data.basement=le.fit_transform(data.basement)
data.hotwaterheating=le.fit_transform(data.hotwaterheating)
data.airconditioning=le.fit_transform(data.airconditioning)
data.parking=le.fit_transform(data.parking)
data.prefarea=le.fit_transform(data.prefarea)
data.furnishingstatus=le.fit_transform(data.furnishingstatus)
data


# In[10]:


x=data.drop('price',axis=1)
y=data.price


# In[11]:


x


# In[12]:


y


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


y_train


# In[17]:


y_test


# In[18]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[19]:


y_pred=model.predict(x_test)
y_pred


# In[20]:


model.intercept_


# In[21]:


model.coef_


# In[22]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print("Mean square error ",mse)


# In[ ]:




