#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


# In[2]:


# Load dataset
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 7_Clustering\\crime_data.csv")
data


# In[4]:


# Normalize the data
def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[5]:


ndata=norm_data(data.iloc[:,1:])
ndata


# In[6]:


# Create dendrogram
dendrogram=sch.dendrogram(sch.linkage(ndata,method='complete'))
dendrogram


# In[8]:


# Create clusters
hc=AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='complete')
hc


# In[9]:


y_hc=hc.fit_predict(ndata)
y_hc


# In[10]:


data['hcluster_id']=y_hc
data


# In[11]:


data.groupby('hcluster_id').mean()


# In[12]:


data['hcluster_id'].value_counts()


# # Using K Means to find out the optimal number of clusters and generating labels accordingly

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[14]:


# Load dataset
data1=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 7_Clustering\\crime_data.csv")
data1


# In[15]:


# Normalize the data
def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[16]:


ndata1 = norm_data(data1.iloc[:,1:])
ndata1


# In[17]:


# We will use the already normalized data for within cluster sum of squares(WCSS)
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,random_state=0)
    kmeans.fit(ndata1)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# ### We can see that optimal number of clusters comes out to be 4

# # Creating clusters

# In[18]:


clusters=KMeans(4,random_state=0)
clusters.fit(ndata1)


# In[19]:


clusters.labels_


# In[20]:


data1['cluster']=clusters.labels_
data1


# In[21]:


data1['cluster'].replace([0,1,2,3],['cluster 0','cluster 1','cluster 2','cluster 3'], inplace = True)
data1


# In[22]:


data1.groupby('cluster').agg('mean').reset_index()


# In[23]:


import seaborn as sns


# In[30]:


data_sort= data1.sort_values('Assault',ascending=False)
data_sort.tail()


# In[32]:


sort=data1.sort_values('Murder', ascending=False)
sort.head()


# In[24]:


sns.lmplot('Murder','Assault', data=data1,hue = 'cluster',fit_reg=False,size=6)


# ### North Carolina has the highest assault rate followed by Florida while Georgia has the highest murder rate.North Dakota is the safest State.
# ### 1) US States belonging to cluster 2 have highest murder rate and are also high on assault rate.
# ### 2) US States belonging to cluster 0 are high on both assault and murder rate.
# ### 3) US States belonging to cluster 3 are the safest with respect to murder and assault rate.
# ### 4) US States belonging to cluster 1 have medicore crime rate with respect to the other States in cluster 0 and cluster 2.

# In[35]:


sns.lmplot('Rape','Assault', data=data1, hue = 'cluster',fit_reg=False)


# In[38]:


data_sort1=data1.sort_values('Rape',ascending=False)
data_sort1.tail()


# ### Nevada has the highest rape rate followed by Alaska while North Dakota is the safest State
# ### 1) Majority of the US States belonging to cluster 0 have highest rape rate and are high on assault rate too.
# ### 2) US States belonging to cluster 3 are the safest with respect to rape and assault rate.
# ### 3) US States belonging to cluster 2 are high on assault and rape rate followed by States in cluster 1.

# In[41]:


sns.lmplot('Rape','Murder',data=data1, hue='cluster',fit_reg=False)


# ### 1) US States belonging to cluster 2 have highest murder rate and are high on rape rate too.
# ### 2) US States belonging to cluster 0 have highest rape rate with high murder rate.
# ### 3) US States belonging to cluster 3 are the safest.

# # Conclusion
# # While US States in cluster 0 and cluster 2 report high crime rate, States belonging to cluster 3 are the safest with States in cluster 1 reporting medicore crime rate.

# In[ ]:




