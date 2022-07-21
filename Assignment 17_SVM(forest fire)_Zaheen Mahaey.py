#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[41]:


# Load Dataset
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 17_SVM\\forestfires.csv")
data


# In[42]:


# Checking for duplicates
data[data.duplicated()]


# In[43]:


df = data.drop_duplicates()
df


# In[45]:


# Visualizing categorical columns
sns.countplot(df['month'])
plt.show()
sns.countplot(df['day'])
plt.show()
sns.countplot(df['size_category'])
plt.show()


# In[6]:


# Dropping columns 'month' and 'day' as we have these columns in binary format
df=df.drop(['month','day'],axis=1)
df


# In[7]:


df.info()


# In[8]:


# Resetting the row index
df=df.reset_index()
df


# In[9]:


df=df.drop('index',axis=1)
df


# In[10]:


# Transforming the categorical variable 'size_category' to numerical form
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df.iloc[:,-1]=labelencoder.fit_transform(df.iloc[:,-1])
df


# In[11]:


# Calculating PPS 
import ppscore as pps


# In[12]:


colnames=list(df.columns)
for i in colnames:
    z=pps.score(data,i,"size_category")
    print('PPS score for {} with size_category is {}'. format(i,z['ppscore']))


# In[13]:


# Dropping the columns with PPS=0 & area
df=df.drop(['DC','temp','monthmay','area'],axis=1)
df


# In[14]:


# Visualizing outliers in the numerical data
fig, ax = plt.subplots(2,3, figsize=(15,10))

sns.boxplot(df.FFMC, ax=ax[0,0])
sns.boxplot(df.DMC, ax=ax[0,1])
sns.boxplot(df.ISI, ax=ax[0,2])
sns.boxplot(df.RH, ax=ax[1,0])
sns.boxplot(df.wind, ax=ax[1,1])
sns.boxplot(df.rain, ax=ax[1,2])

plt.tight_layout()
plt.show()


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled=scaler.fit_transform(df)
scaled


# In[16]:


X=scaled[:,:-1]
Y=scaled[:,-1]


# In[17]:


X.shape


# In[18]:


Y.shape


# In[24]:


x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.3, random_state=7)


# # GridSearchCV

# In[29]:


clf=SVC()
param_grid=[{'kernel': ['rbf'],'gamma': [100,50,20,10,1,0.5,0.1,0.01,0.001,0.0001],
            'C':[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0.5,0.1,0.01,0.001,0.0001]}]
gsv=GridSearchCV(clf,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[30]:


gsv.best_params_,gsv.best_score_


# In[33]:


clf = SVC(gamma=50,C=0.5)
clf.fit(x_train,y_train)
y_preds_train= clf.predict(x_train)
acc=accuracy_score(y_train,y_preds_train)*100
print("accuracy=",acc)
print(confusion_matrix(y_train,y_preds_train))


# In[34]:


y_preds_test=clf.predict(x_test)
acc=accuracy_score(y_test,y_preds_test)*100
print("accuracy=",acc)
print(confusion_matrix(y_test,y_preds_test))


# ### We can see that testing accuracy is greater than training accuracy. Hence, applying Stratified K Fold validation technique separately to cross check the accuracy.

# In[38]:


from sklearn.model_selection import StratifiedKFold
clf = SVC(gamma=50,C=0.5)
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(clf, X, Y, cv=cv)
print(np.mean(scores))


# ### Thus, we get overall average accuracy of 73.5%

# In[ ]:




