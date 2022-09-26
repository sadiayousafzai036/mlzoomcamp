#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[34]:


numcols = [
'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value',
'ocean_proximity']
data = pd.read_csv('housing.csv')


# In[7]:


data.head(10)


# In[8]:


data = data[cols]


# In[9]:


data.isnull().sum()


# In[10]:


data['total_bedrooms']=data.total_bedrooms.fillna(0)


# In[13]:


data['rooms_per_household'] = data['total_rooms'] / data['households'] 
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms'] 
data['population_per_household'] = data['population'] / data['households'] 


# In[15]:


data.describe(include=["O"])


# In[17]:


data.dtypes


# In[18]:


data_numeric = data.copy()
data_numeric = data.drop(['ocean_proximity'], axis=1)
data_numeric.describe()


# In[23]:


pd.options.display.max_rows = 4000
data_numeric.corr().unstack().sort_values(ascending = False)


# In[24]:


data_class = data.copy()
mean = data_class['median_house_value'].mean()

data_class['above_average'] = np.where(data_class['median_house_value']>=mean,1,0)


# In[25]:


data_class = data_class.drop('median_house_value', axis=1)


# In[26]:


from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(data_class, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
#########################################
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
######################################
y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values


# In[27]:


from sklearn.metrics import mutual_info_score
cat = ['ocean_proximity']
def calculate_mi(series):
    return mutual_info_score(series, df_train.above_average)

df_mi = df_train[cat].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
df_mi


# In[31]:


num = [
'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income']
#data = pd.read_csv('housing.csv')
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_dict = df_train[cat + num].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict(X_val)

accuracy = np.round(accuracy_score(y_val, y_pred),2)
print(accuracy)


# In[33]:


features = cat + num
orig_score = accuracy


for c in features:
    subset = features.copy()
    subset.remove(c)
    
    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    score = accuracy_score(y_val, y_pred)
    print(c, orig_score - score, score)


# In[36]:


data['median_house_value']=np.log1p(data['median_house_value'])
df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values
del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# In[37]:


train_dict = df_train[cat + num].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a,random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(a, round(score, 3))


# In[ ]:




