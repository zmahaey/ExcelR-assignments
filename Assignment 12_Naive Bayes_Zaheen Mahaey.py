#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load train data
file = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 12_Naive Bayes\\SalaryData_Train.csv")
file


# In[3]:


file.info()


# In[4]:


file.shape


# In[5]:


# Applying labelencoder to train data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
file.iloc[:,1]=labelencoder.fit_transform(file.iloc[:,1])
file.iloc[:,2]=labelencoder.fit_transform(file.iloc[:,2])
file.iloc[:,4]=labelencoder.fit_transform(file.iloc[:,4])
file.iloc[:,5]=labelencoder.fit_transform(file.iloc[:,5])
file.iloc[:,6]=labelencoder.fit_transform(file.iloc[:,6])
file.iloc[:,7]=labelencoder.fit_transform(file.iloc[:,7])
file.iloc[:,8]=labelencoder.fit_transform(file.iloc[:,8])
file.iloc[:,12]=labelencoder.fit_transform(file.iloc[:,12])
file.iloc[:,13]=labelencoder.fit_transform(file.iloc[:,13])


# In[6]:


file


# In[7]:


from sklearn.naive_bayes import CategoricalNB
clf=CategoricalNB()


# In[8]:


X=file.iloc[:,:-1]
Y=file.iloc[:,-1]


# In[9]:


clf.fit(X,Y)


# In[10]:


# Loading test data
file_test=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 12_Naive Bayes\\SalaryData_Test.csv")
file_test


# In[11]:


file_test.info()


# In[12]:


test=file_test.copy()


# In[13]:


test.iloc[:,1]=labelencoder.fit_transform(test.iloc[:,1])
test.iloc[:,2]=labelencoder.fit_transform(test.iloc[:,2])
test.iloc[:,4]=labelencoder.fit_transform(test.iloc[:,4])
test.iloc[:,5]=labelencoder.fit_transform(test.iloc[:,5])
test.iloc[:,6]=labelencoder.fit_transform(test.iloc[:,6])
test.iloc[:,7]=labelencoder.fit_transform(test.iloc[:,7])
test.iloc[:,8]=labelencoder.fit_transform(test.iloc[:,8])
test.iloc[:,12]=labelencoder.fit_transform(test.iloc[:,12])
test.iloc[:,13]=labelencoder.fit_transform(test.iloc[:,13])


# In[14]:


test


# In[15]:


# Dropping column Salary
test_new=test.drop('Salary',axis=1)
test_new


# In[16]:


test_new_val=test_new.values
test_new_val


# In[17]:


# Predicting salary from Naive Bayes on test data
prediction_salary= clf.predict(test_new_val)
prediction_salary


# # Confusion Matrix

# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test['Salary'],prediction_salary)
print(cm)


# In[19]:


test['Salary'].value_counts()


# In[20]:


TN=10475
TP=2426
FP=885
FN=1274


# # Sensitivity

# In[21]:


Sensitivity=(TP/(TP+FN))*100
Sensitivity


# # Specificity

# In[22]:


specificity=(TN/(TN+FP))*100
specificity


# # Precision

# In[23]:


Precision = (TP/(TP+FP))*100
Precision


# # F score

# In[24]:


F_score= (2*Precision*Sensitivity)/(Precision+Sensitivity)
F_score


# # Classification report

# In[25]:


from sklearn.metrics import classification_report
print(classification_report(test['Salary'],prediction_salary))


# In[26]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[27]:


# ROC Curve
fpr,tpr,thresholds = roc_curve(test['Salary'],prediction_salary)
auc=roc_auc_score(test['Salary'],prediction_salary)
print(auc)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr, color='red', label = 'logit model (area=%0.2f)'%auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False positive rate or [1-True negative rate]')
plt.ylabel('Ture Positive Rate')

