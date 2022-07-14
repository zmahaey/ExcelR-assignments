#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


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
df['target']=pd.cut(x=df['Sales'], bins=[0,8.0,17], right=False, labels=['low','high'])
df


# In[7]:


df['target'].value_counts()


# In[8]:


df=df.drop('Sales',axis=1)
df.head()


# In[9]:


# Converting categorical column values to numerical values through label encoder
labelencoder=LabelEncoder()
df.iloc[:,5]=labelencoder.fit_transform(df.iloc[:,5])
df.iloc[:,8]=labelencoder.fit_transform(df.iloc[:,8])
df.iloc[:,9]=labelencoder.fit_transform(df.iloc[:,9])
df.iloc[:,10]=labelencoder.fit_transform(df.iloc[:,10])


# In[10]:


df.head()


# In[11]:


# Visualizing the data
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue='target',palette='tab10')
plt.show()


# In[11]:


X=df.iloc[:,0:10]
Y=df.iloc[:,10]


# # Applying Random Forest algorithm and calculating the accuracy

# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=7)


# In[21]:


model= RandomForestClassifier(n_estimators=1000, max_features=3)
model.fit(x_train,y_train)


# In[22]:


preds=model.predict(x_test)
preds


# In[23]:


from sklearn.metrics import confusion_matrix


# In[24]:


cm=confusion_matrix(y_test, preds)
print(cm)


# In[25]:


TN=19
TP=41
FP=13
FN=7


# In[26]:


# Sensitivity
Sensitivity = TP/(TP+FN)
print(Sensitivity*100)


# In[28]:


# Specificity
Specificity = TN/(TN+FP)
print(Specificity*100)


# In[29]:


# Precision
Precision = TP/(TP+FP)
print(Precision*100)


# In[31]:


# F Score
F_score=(2*Precision*Sensitivity)/(Precision+Sensitivity)
print(F_score*100)


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,preds))


# In[34]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[35]:


# ROC Curve
fpr,tpr,thresholds = roc_curve(y_test,preds)
auc=roc_auc_score(y_test,preds)
print(auc)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr, color='red', label = 'logit model (area=%0.2f)'%auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False positive rate or [1-True negative rate]')
plt.ylabel('Ture Positive Rate')


# In[ ]:




