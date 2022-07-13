#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Load the data
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 14_Decision Trees\\Fraud_check.csv")
df


# In[3]:


df.info()


# In[4]:


#Checking duplicates
df[df.duplicated()]


# In[5]:


# Checking maximum value in column "Taxable.Income"
df['Taxable.Income'].max()


# In[6]:


# Creating a column "risk" & treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

df['risk']=pd.cut(x=df['Taxable.Income'],bins=[0,30000,100000],right=True, labels=['risky','good'])
df


# In[7]:


df['risk'].value_counts()


# In[8]:


data=df.copy()
data.head()


# In[9]:


data=data.drop('Taxable.Income',axis=1)
data.head()


# In[10]:


# making use of label encoder for categorical columns
labelencoder=LabelEncoder()
data.iloc[:,0]=labelencoder.fit_transform(data.iloc[:,0])
data.iloc[:,1]=labelencoder.fit_transform(data.iloc[:,1])
data.iloc[:,5]=labelencoder.fit_transform(data.iloc[:,5])
data.iloc[:,4]=labelencoder.fit_transform(data.iloc[:,4])


# In[11]:


data


# In[12]:


# Visualizing features through pairplot
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data,hue='risk',palette='tab10')
plt.show()


# In[13]:


X=data.iloc[:,0:5]
Y=data.iloc[:,5]


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# In[15]:


# Building the model
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,class_weight='balanced')
model.fit(x_train,y_train)


# In[16]:


colnames=list(data.columns)
colnames


# In[17]:


# Plotting the tree
fn=['Undergrad','Marital.Status','City.Population','Work.Experience','Urban']
cn=['good','risky']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
tree.plot_tree(model,feature_names=fn, class_names=cn,filled=True)


# In[18]:


preds=model.predict(x_test)
preds


# In[19]:


pd.crosstab(preds,y_test)


# In[20]:


np.mean(preds==y_test)


# ### We got a very low accuracy of 30% which could be improved.

# # Applying Bagging ensemble technique to improve the accuracy

# In[21]:


from sklearn.ensemble import BaggingClassifier


# In[22]:


# Bagging
cart=DecisionTreeClassifier()
model=BaggingClassifier(base_estimator=cart,n_estimators=1000,random_state=7)
model.fit(x_train,y_train)


# In[23]:


model.score(x_train,y_train)


# In[24]:


model.score(x_test,y_test)


# ### We can see that the accuracy of the model has improved from 30% to 72.5% by applying the bagging ensemble technique

# In[ ]:




