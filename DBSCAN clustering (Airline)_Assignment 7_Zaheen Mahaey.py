#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics


# In[2]:


data=pd.read_excel(r"C:\Users\zahee\Desktop\Data Science\Assignments\Assignment 7_Clustering\\EastWestAirlines.xlsx",
                  sheet_name='data')
data.head()


# In[3]:


# Normalizing the data
def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[4]:


ndata=norm_data(data.iloc[:,1:])
ndata.head()


# In[45]:


dbscan = DBSCAN(eps=0.4,min_samples=8)
dbscan.fit(ndata)


# In[46]:


dbscan.labels_


# In[47]:


data['cluster']=dbscan.labels_
data.head()


# In[48]:


data['cluster'].value_counts()


# In[49]:


array=ndata.values
array


# In[50]:


score=metrics.silhouette_score(array,dbscan.labels_)
score


# In[59]:


# Plotting the graph
def dbscan_new(array,eps,min_samples):
    db=DBSCAN(eps=eps,min_samples=min_samples)
    db.fit(array)
    y_pred=db.fit_predict(array)
    plt.scatter(array[:,0],array[:,1],c=y_pred,cmap='Paired')


# In[60]:


dbscan_new(array,0.4,8)


# In[62]:


import seaborn as sns
sns.lmplot('Balance','Qual_miles',data=data, hue='cluster', fit_reg=False)


# In[ ]:




