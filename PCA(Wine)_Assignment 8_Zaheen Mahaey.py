#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[4]:


# Load dataset
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 8_PCA\\wine.csv")
data.head()


# In[9]:


# Excluding the first column of the data
df=data.iloc[:,1:]
df.head()


# In[10]:


#converting into array
df_array= df.values
df_array


# In[11]:


# Normalizing the data
normal_df= scale(df_array)
normal_df


# In[12]:


pca=PCA(n_components=13)
pca_values=pca.fit_transform(normal_df)


# In[14]:


var=pca.explained_variance_ratio_
var


# In[15]:


var1= np.cumsum((var)*100)
var1


# In[16]:


pca.components_


# In[18]:


plt.plot(var1,color='red')


# In[19]:


x=pca_values[:,0]
y=pca_values[:,1]
z=pca_values[:,2]


# In[24]:


wine=pd.DataFrame(pca_values[:,0:3],columns=['pc1','pc2','pc3'])
wine


# In[26]:


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(wine["pc1"], wine["pc2"], wine["pc3"])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


# # Applying hierarchical clustering

# In[27]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[28]:


dendrogram=sch.dendrogram(sch.linkage(wine,method='complete'))
dendrogram


# In[29]:


hc=AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
hc


# In[31]:


y_pred=hc.fit_predict(wine)
y_pred


# In[32]:


wine1=wine.copy()


# In[33]:


wine1['cluster']=y_pred
wine1.head()


# In[34]:


wine1['cluster'].value_counts()


# In[35]:


wine1.groupby(wine1['cluster']).mean()


# # Applying K- means clustering

# In[36]:


from sklearn.cluster import KMeans


# In[37]:


wine2=wine.copy()


# In[41]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, random_state=0)
    kmeans.fit(wine2)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


# # Thus, optimal number of clusters is 3

# In[42]:


kmeans_new=KMeans(n_clusters=3, random_state=0)
kmeans_new.fit(wine2)


# In[43]:


kmeans_new.labels_


# In[44]:


wine2['cluster']=kmeans_new.labels_
wine2.head()


# In[58]:


wine3=pd.concat([data,wine2],axis=1)
wine3.head()


# In[59]:


# Plotting the first two components since they explain more variance than the third component
import seaborn as sns
sns.scatterplot(wine3['pc1'],wine3['pc2'],data=wine3, hue=wine3['cluster'], palette=['g','r','c'])# palette=g,r,c,m


# In[ ]:




