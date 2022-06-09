#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN


# In[3]:


# Load dataset
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 7_Clustering\\crime_data.csv")
df


# In[4]:


data=df.copy()


# In[5]:


data=data.drop('Unnamed: 0', axis =1)
data.head()


# In[6]:


def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[7]:


ndata = norm_data(data.iloc[:,0:])
ndata.head()


# In[10]:


array=ndata.values
array


# In[165]:


dbscan=DBSCAN(eps=0.35,min_samples=8)
dbscan.fit(array)


# In[166]:


labels = dbscan.labels_
labels


# In[167]:


df['cluster']=pd.DataFrame(labels)
df.head()


# In[168]:


df['cluster'].value_counts()


# In[169]:


X=metrics.silhouette_score(array,dbscan.labels_)
X


# In[143]:


def new_dbscan(array,eps,min_samples):
    db=DBSCAN(eps=eps,min_samples=min_samples)
    db.fit(array)
    y_pred=db.fit_predict(array)
    plt.scatter(array[:,0],array[:,1], c=y_pred, cmap='Paired')
    plt.title('DBSCAN')


# In[171]:


new_dbscan(array,0.35,8)


# In[172]:


import seaborn as sns


# In[173]:


sns.lmplot('Murder','Assault', data=df, hue='cluster',fit_reg=False)


# # Since this is not as dense a dataset and has very low noise, K-means would be a better alternative for analysis

# In[ ]:




