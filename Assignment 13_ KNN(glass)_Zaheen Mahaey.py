#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[20]:


# Load Dataset
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 13_KNN\\glass.csv")
df


# In[21]:


df.info()


# In[22]:


# Checking for duplicates
df[df.duplicated()]


# In[23]:


# Dropping duplicates
data=df.drop_duplicates()
data


# In[24]:


# Checking for the independence of all the features
import seaborn as sns
sns.heatmap(df.corr(),annot=True, cmap='tab10',linecolor='w',linewidth = 2)


# ### We notice that Ca does not affect the "Type". Also Ca and RI are highly correlated.Hence, we can drop Ca

# In[25]:


data=data.drop('Ca',axis=1)
data.head()


# In[26]:


data.shape


# In[31]:


# Plotting pairplot to have a look into features
import matplotlib.pyplot as plt
sns.pairplot(data, hue='Type',palette='tab10')
plt.show()


# ### Thus we can see that the data is non linear

# In[32]:


# Preparing for normalizing the data
array=data.values
array


# In[33]:


X=array[:,0:8]
Y=array[:,8]


# In[34]:


X.shape


# In[35]:


# Normalizing the data
scale=StandardScaler()
scaled_data=scale.fit_transform(X)


# In[36]:


X


# In[37]:


Y


# # Applying KNN model

# In[38]:


kfold=KFold(n_splits=5)


# In[39]:


model=KNeighborsClassifier(n_neighbors=10)
result=cross_val_score(model,X,Y,cv=kfold)


# In[40]:


print(result.mean())


# ### Thus we see an accuracy of 22.41% with the neighbors as 10 in the model

# # Performing Algorithm tuning by choosing optimal value of K

# In[45]:


n_neighbors=np.array(range(1,51))
param_grid=dict(n_neighbors=n_neighbors)
param_grid


# In[46]:


KNN=KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN, param_grid=param_grid)
grid.fit(X,Y)


# In[47]:


print(grid.best_score_)
print(grid.best_params_)


# ### The best results are obtained with k=1 and the accuracy achieved is 64%

# # Visualizing the result

# In[48]:


k_range = range(1,51)
k_score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score= cross_val_score(knn,X,Y,cv=5)
    k_score.append(score.mean())
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_range,k_score)
plt.xlabel('Values of K for KNN')
plt.ylabel('Cross Validated Accuracy')
plt.show()


# In[ ]:




