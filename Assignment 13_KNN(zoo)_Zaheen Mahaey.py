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


# Plotting correlation among features since Pearson correlation can be used for categorical variables also but 
# encoded as 0/1 only 
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,10))
cor=zoo.corr()
sns.heatmap(cor, annot=True,linecolor='w',linewidth=2,cmap='tab10')
plt.show()


# ### Thus we see high correlation between 1) hair and milk 2) hair and eggs 3) Milk and eggs. Which feature to drop can be determined by feature selection as done below

# In[5]:


#Dropping unnecessary column 'animal name'
zoo=zoo.drop('animal name',axis=1)
zoo.head()


# In[6]:


array=zoo.values
array


# In[7]:


X=array[:,0:16]
Y=array[:,16]


# In[8]:


X


# In[9]:


Y


# # Calculating feature importance using Random Forest 

# ### Using Random forest algorithm, the feature importance can be measured as the average impurity decrease computed from all decision trees in the forest.

# In[11]:


from sklearn.ensemble import RandomForestClassifier


# In[12]:


model=RandomForestClassifier()
model.fit(X,Y)


# In[16]:


# get importance
importance = model.feature_importances_
importance


# In[39]:


# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: {}, Score: {}' .format(i,round(v,5)))


# In[40]:


# plot feature importance 
import matplotlib.pyplot as plt
plt.bar(range(len(importance)), importance)
plt.show()


# ### Since 'hair' column  has low feature score in predciting the 'Y', so dropping it & retaining 'milk' and 'eggs' column due to relatively high feature score

# In[22]:


zoo=zoo.drop('hair',axis=1)
zoo


# In[26]:


x=zoo.iloc[:,0:15]
y=zoo.iloc[:,15]


# # Applying KNN model using k-fold cross validation

# In[30]:


kfold= KFold(n_splits=4)


# In[31]:


model=KNeighborsClassifier(n_neighbors=4)
result= cross_val_score(model,x,y,cv=kfold)


# In[32]:


print(result.mean())


# ### Thus we see an accuracy of 86% with the neighbors as 4 in the model

# # Grid search for algorithm tuning

# In[33]:


n_neighbors= np.array(range(1,51))
param_grid=dict(n_neighbors=n_neighbors)
param_grid


# In[34]:


KNN= KNeighborsClassifier()
grid=GridSearchCV(estimator=KNN,param_grid=param_grid)
grid.fit(x,y)


# In[35]:


print(grid.best_score_)
print(grid.best_params_)


# ### The best results are obtained with k=1 and the accuracy achieved is 95%

# # Visualizing the results

# In[36]:


k_range= range(1,51)
k_score = []

for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    score= cross_val_score(knn,x,y,cv=4)
    k_score.append(score.mean())
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_range,k_score)
plt.xlabel("Values of K for KNN")
plt.ylabel("Cross Validated Accuracy")
plt.show()


# In[ ]:




