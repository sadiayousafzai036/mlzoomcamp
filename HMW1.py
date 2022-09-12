#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


np.__version__


# In[22]:


df_7 = pd.read_csv("/home/ubuntu/mlzoomcamprepo/data.csv")


# In[13]:


df.shape


# In[14]:


df.groupby('Make').size().nlargest(3)


# In[16]:


df_audi = df[df.Make == 'Audi']
df_audi['Model'].unique().shape


# In[17]:


df.isnull().sum()


# In[21]:


df['Engine Cylinders'].median() #6.0
df['Engine Cylinders'].mode() #4
df['Engine Cylinders'].fillna(value = 4.0, inplace = True)
df['Engine Cylinders'].median()


# In[28]:


df_lotus = df_7[df_7['Make'] == 'Lotus']
df_lotus = df_lotus[['Engine HP', 'Engine Cylinders']]


# In[30]:


df_lotus = df_lotus.drop_duplicates()


# In[31]:


df_lotus


# In[49]:


X = df_lotus.to_numpy()
T = X.dot(X.transpose())


# In[50]:


I = np.linalg.inv(T)


# In[51]:


y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])


# In[53]:


I.dot(T).dot(y)


# In[ ]:




