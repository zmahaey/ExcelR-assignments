#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


# loading Excel file (r is used to to address special character, such as '\')
data=pd.read_excel(r"C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 7_Clustering\\EastWestAirlines.xlsx",
                  sheet_name='data')
data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.shape


# In[6]:


# Normalizing the data
def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[7]:


ndata= norm_data(data.iloc[:,1:])
ndata.head()


# In[8]:


dendrogram = sch.dendrogram(sch.linkage(ndata,method='complete'))
dendrogram


# In[24]:


hc=AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='complete')
hc


# In[25]:


y_pred=hc.fit_predict(ndata)
y_pred


# In[26]:


data['cluster_id']=y_pred
data.head()


# In[27]:


data['cluster_id'].value_counts()


# In[14]:


# Dropping ID column
data=data.drop('ID#',axis=1)


# In[15]:


data.groupby(data['cluster_id']).mean()


# In[ ]:




