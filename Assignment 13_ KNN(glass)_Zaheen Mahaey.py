#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load Dataset
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 13_KNN\\glass.csv")
df


# In[3]:


array=df.values
array


# In[4]:


X=array[:,0:9]
Y=array[:,9]


# In[5]:


X.shape


# In[6]:


scale=StandardScaler()
scaled_data=scale.fit_transform(X)


# In[7]:


X


# In[8]:


Y


# # Applying KNN model

# In[9]:


kfold=KFold(n_splits=5)


# In[10]:


model=KNeighborsClassifier(n_neighbors=10)
result=cross_val_score(model,X,Y,cv=kfold)


# In[11]:


print(result.mean())


# ### Thus we see an accuracy of 27.45% with the neighbors as 10 in the model

# # Performing Algorithm tuning by choosing optimal value of K

# In[17]:


n_neighbors=np.array(range(1,81))
param_grid=dict(n_neighbors=n_neighbors)
param_grid


# In[18]:


KNN=KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN, param_grid=param_grid)
grid.fit(X,Y)


# In[19]:


print(grid.best_score_)
print(grid.best_params_)


# ### The best results are obtained with k=1 and the accuracy achieved is 64%

# # Visualizing the result

# In[21]:


k_range = range(1,81)
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




