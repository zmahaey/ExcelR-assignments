#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


# Load dataset from excel file
data=pd.read_excel(r"C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 7_Clustering\\EastWestAirlines.xlsx",
                  sheet_name='data')
data.head()


# In[5]:


# Normalizing the data
def norm_data(i):
    return(i-i.min())/(i.max()-i.min())


# In[6]:


ndata=norm_data(data.iloc[:,1:])
ndata.head()


# In[7]:


# Determining the optimum number of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, random_state=0)
    kmeans.fit(ndata)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# # Hence, number of optimal clusters obtained are 4

# In[8]:


km = KMeans(n_clusters=4, random_state=0)
km.fit(ndata)


# In[9]:


km.labels_


# In[10]:


data['cluster']=km.labels_
data.head()


# In[12]:


data['cluster'].value_counts()


# In[13]:


import seaborn as sns


# In[33]:


sns.scatterplot(data['Days_since_enroll'], data['Bonus_miles'], hue=data['cluster'],palette='Spectral')
plt.show()


# ### Customers in cluster 2 (mainly) and cluster 3 should be made aware of the number of miles earned from non-flight bonus transactions in the past 12 months which could result in increase in revenue 

# In[52]:


sns.barplot(data['cluster'], data['Flight_trans_12'])


# ### 1)In the past 12 months, customers in cluster 0 have travelled the most followed by cluster 2 customers.
# ### 2)In the past 12 months, customers in cluster 1 and cluster 3 have travelled the least.

# In[36]:


sns.barplot(data['cluster'], data['Flight_miles_12mo'])


# ### In the past 12 months, customers belonging to cluster 0 have the most flight miles followed by customers in cluster 2 (these customers should be made aware about the redemption of flight credit miles to increase revenue). Customers in cluster 1 and cluster 3 have the least filght miles.

# In[46]:


sns.scatterplot(data['Balance'],data['Qual_miles'], hue=data['cluster'],palette='Spectral')


# ### There are many eligible customers in all the clusters who can redeem miles from their balance and upgrade to topflight status

# In[49]:


sns.scatterplot(data['cc1_miles'],data['Flight_trans_12'], hue=data['cluster'],palette='Spectral')


# ### 1) For the past 12 months, frequent flier program is the most popular amongst cluster 2 customers and least popular amongst cluster 3 customers.
# ### 2) Customers in cluster 0 and cluster 1 should be encouraged more to travel through redemption of frequent flier credits.

# In[50]:


sns.scatterplot(data['cc2_miles'], data['Flight_trans_12'], hue=data['cluster'],palette='Spectral')


# ### 1) Rewards credit mile program is the least popular in customers belonging to cluster 3 in the past 12 months.
# ### 2) This program needs to be popularized as only handful of customers have earned greater than 5000 miles in the past 12 months.

# In[51]:


sns.scatterplot(data['cc3_miles'], data['Flight_trans_12'], hue=data['cluster'],palette='Spectral')


# ### 1) Small business credit card program is the least popular of all the programs although a few people use it for earning credit miles.

# # Reward programs summary:
# ### 1)Customers belonging to cluster 3 need to be targeted for the usage of rewards program.
# ### 2)Customers in cluster 0 and cluster 1 should be encouraged to redeem their credit miles thereby resulting in an increase in revenue.
# ### 3)Most popular reward program is the frequent flier program followed by not so popular programs - rewards credit card and small business credit card respectively.
# 

# In[ ]:




