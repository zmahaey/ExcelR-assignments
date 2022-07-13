#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


#Load dataset
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 14_Decision Trees\\Company_Data.csv")
df


# In[3]:


df.info()


# In[4]:


# Checking for duplicate rows
df[df.duplicated()]


# In[5]:


df['Sales'].max()


# In[6]:


# Creating column 'target' and treating those who have sales >8 as high and sales <=8 as low
df['target']=pd.cut(x=df['Sales'], bins=[0,8,17], right=True, labels=['low','high'])
df


# In[7]:


df['target'].value_counts()


# In[8]:


df=df.drop('Sales',axis=1)
df.head()


# In[9]:


# Converting categorical column values to continuous data through label encoder
labelencoder=LabelEncoder()
df.iloc[:,5]=labelencoder.fit_transform(df.iloc[:,5])
df.iloc[:,8]=labelencoder.fit_transform(df.iloc[:,8])
df.iloc[:,9]=labelencoder.fit_transform(df.iloc[:,9])
df.iloc[:,10]=labelencoder.fit_transform(df.iloc[:,10])


# In[10]:


df.head()


# In[24]:


# Visualizing features once for all
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue='target',palette='tab10')
plt.show()


# In[11]:


X=df.iloc[:,0:10]
Y=df.iloc[:,10]


# In[12]:


colnames=list(df.columns)
colnames


# In[13]:


# Splitting the data into test and train data
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=7)


# In[14]:


# Building the Decision Tree
model=DecisionTreeClassifier(criterion='entropy',max_depth =3)
model.fit(x_train,y_train)


# In[15]:


#Plotting the tree
fn=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
cn=['high','low']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)

tree.plot_tree(model,feature_names=fn, class_names=cn,filled =True)


# In[16]:


preds=model.predict(x_test)
preds


# In[17]:


pd.crosstab(y_test,preds)


# In[18]:


np.mean(y_test==preds)


# ### The accuracy of the model is 71.25% which could be improved further

# # Applying Bagging ensemble technique to improve the accuracy

# In[19]:


from sklearn.ensemble import BaggingClassifier


# In[20]:


# Bagging
cart=DecisionTreeClassifier()
model=BaggingClassifier(base_estimator=cart,n_estimators=800,random_state=7)
model.fit(x_train,y_train)


# In[21]:


model.score(x_train,y_train)


# In[22]:


model.score(x_test,y_test)


# ### Thus we see that the accuracy of the model has increased from 71.2% to 75% by the use of ensemble technique.

# In[ ]:




