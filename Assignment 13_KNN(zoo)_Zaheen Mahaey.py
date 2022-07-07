#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


#Load Dataset
zoo=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 13_KNN\\Zoo.csv")
zoo


# In[3]:


zoo.info()


# In[4]:


zoo=zoo.drop('animal name',axis=1)
zoo.head()


# In[5]:


array=zoo.values
array


# In[6]:


X=array[:,0:16]
Y=array[:,16]


# In[7]:


X


# In[8]:


Y


# # Applying KNN model using k-fold cross validation

# In[17]:


kfold= KFold(n_splits=4)


# In[18]:


model=KNeighborsClassifier(n_neighbors=4)
result= cross_val_score(model,X,Y,cv=kfold)


# In[19]:


print(result.mean())


# ### Thus we see an accuracy of 86% with the neighbors as 4 in the model

# # Grid search for algorithm tuning

# In[25]:


n_neighbors= np.array(range(1,51))
param_grid=dict(n_neighbors=n_neighbors)
param_grid


# In[26]:


KNN= KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN,param_grid=param_grid)
grid.fit(X,Y)


# In[27]:


print(grid.best_score_)
print(grid.best_params_)


# ### The best results are obtained with k=1 and the accuracy achieved is 97%

# # Visualizing the results

# In[28]:


k_range= range(1,51)
k_score = []

for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    score= cross_val_score(knn,X,Y,cv=4)
    k_score.append(score.mean())
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_range,k_score)
plt.xlabel("Values of K for KNN")
plt.ylabel("Cross Validated Accuracy")
plt.show()


# In[ ]:




