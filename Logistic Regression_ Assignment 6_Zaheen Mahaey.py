#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


#Loading the data
data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 6_Logistic regression\\bank-full.csv")
data


# In[5]:


Test = data.columns
Test


# In[6]:


array=data.values
array


# In[7]:


import numpy as np
array.shape


# In[8]:


import string as str


# In[9]:


# Storing the "array" in a dataframe
z=pd.DataFrame(array,columns=['test'])
z


# In[10]:


# Removing double quotes from the text.This function is only applicable to column in dataframe and not dataframe as
# a whole
u=z['test'].str.replace('"','')
u


# In[11]:


b=pd.DataFrame(u)
b


# In[12]:


# Splitiing the column into 17 columns with semi colon as delimiter
zaheen1 = b['test'].str.split(pat=';',expand=True)
zaheen1


# In[13]:


# Renaming the column axis
zaheen1=zaheen1.set_axis(["age","job","marital","education","default","balance","housing","loan","contact","day",
                 "month","duration","campaign","pdays","previous","poutcome","y"],axis=1)
zaheen1


# In[14]:


data_cleaned = pd.DataFrame(zaheen1)
data_cleaned


# In[15]:


data1=data_cleaned.copy()
data1.head()


# In[16]:


data1.info()


# In[17]:


data1.isnull().sum()


# In[18]:


data1['job'].value_counts()


# In[19]:


# Grouping various job categories into one category employed
data1['job'].replace(['blue-collar','management','technician','admin.','services','self-employed',
                     'entrepreneur','housemaid'],'epmloyed',inplace=True)


# In[20]:


data1['job'].value_counts()


# In[21]:


data1['month'].value_counts()


# In[22]:


# Replacing the month names with their corresponding values
data1['month'].replace(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                     [1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)


# In[23]:


data1['month'].value_counts()


# In[24]:


#Changing the dtype of column pdays from object to int
data1['pdays']=pd.to_numeric(data1['pdays'])


# In[25]:


data1[data1['pdays']==0]


# In[26]:


# Assigning all the -1 values in pdays to 0
data1['pdays'].replace(-1,0,inplace=True)


# In[27]:


data1['pdays'].value_counts()


# In[28]:


data1[data1['pdays']==0]


# In[29]:


data1['default'].value_counts()


# In[30]:


# Replacing yes with 1 and no with 0 in default column
data1['default'].replace(['no','yes'],[0,1],inplace = True)


# In[31]:


data1['default'].value_counts()


# In[32]:


data1['housing'].value_counts()


# In[33]:


data1['housing'].replace(['no','yes'],[0,1],inplace=True)


# In[34]:


data1['housing'].value_counts()


# In[35]:


data1['loan'].value_counts()


# In[36]:


# Replacing yes with 1 and no with 0 in loan column
data1['loan'].replace(['no','yes'],[0,1],inplace=True)


# In[37]:


data1['loan'].value_counts()


# In[38]:


data1['y'].value_counts()


# In[39]:


data1['y'].replace(['no','yes'],[0,1], inplace=True)


# In[40]:


data1['y'].value_counts()


# In[110]:


data1.head()


# In[42]:


data1.info()


# In[43]:


# Assigning dtypes to various columns
data1['age']=pd.to_numeric(data1['age'])
data1['job']=data1['job'].astype('category')
data1['marital']=data1['marital'].astype('category')
data1['education']=data1['education'].astype('category')
data1['balance']=pd.to_numeric(data1['balance'])
data1['contact']=data1['contact'].astype('category')
data1['day']=pd.to_numeric(data1['day'])
data1['duration']=pd.to_numeric(data1['duration'])
data1['campaign']=pd.to_numeric(data1['campaign'])
data1['previous']=pd.to_numeric(data1['previous'])
data1['poutcome']=data1['poutcome'].astype('category')


# In[44]:


data1.info()


# In[45]:


# Creating dummy variables for job
job_d=pd.get_dummies(data1['job'],prefix='job')
data2=data1.join(job_d)
data2


# In[46]:


# Creating dummy variable for "marital" column
marital_d=pd.get_dummies(data2['marital'], prefix='mar')
data3=data2.join(marital_d)
data3.head()


# In[47]:


# Dropping columns "job, marital and contact( as contact is unnecessary column)"
data3=data3.drop(["job","marital","contact"],axis=1)
data3.head()


# In[48]:


education_d=pd.get_dummies(data3['education'],prefix='edu')
data4=data3.join(education_d)
data4=data4.drop('education',axis=1)
data4.head()


# In[49]:


poutcome_d=pd.get_dummies(data4['poutcome'],prefix = 'pout')
data5=data4.join(poutcome_d)
data5=data5.drop('poutcome',axis=1)
data5.head()


# In[50]:


data5.info()


# In[51]:


# Rearranging column "y" to index 0 with the new name "term_subscription"
output=data5['y']
data5.insert(loc=0, column='term_subscription', value=output)


# In[52]:


# Validating the rearrangement of the column "y"
data5[data5['term_subscription']==data5['y']]


# In[53]:


# Dropping column "y"
data5=data5.drop('y',axis=1)
data5.head()


# # Applying Logistic Regression on dataset "data5"

# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


# Dividing the data into input and output variable
X=data5.iloc[:,1:]
Y=data5.iloc[:,0]


# In[56]:


classifier=LogisticRegression()
classifier.fit(X,Y)


# In[57]:


classifier.coef_


# In[58]:


classifier.predict_proba(X)


# In[59]:


classifier.score(X,Y)


# In[60]:


classifier.predict(X)


# In[61]:


y_pred=classifier.predict(X)


# In[62]:


y_pred_df=pd.DataFrame({"Actual":Y,"Predictions":y_pred})
y_pred_df


# # Confusion matrix

# In[63]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y,y_pred)
print(cm)


# In[64]:


TP= len(y_pred_df[(y_pred_df['Actual']==1) & (y_pred_df['Predictions']==1)])
TN= len(y_pred_df[(y_pred_df['Actual']==0) & (y_pred_df['Predictions']==0)])
FP= len(y_pred_df[(y_pred_df['Actual']==0) & (y_pred_df['Predictions']==1)])
FN= len(y_pred_df[(y_pred_df['Actual']==1) & (y_pred_df['Predictions']==0)])


# In[65]:


print('True Positives', TP)
print('True Negatives', TN)
print('False Positives', FP)
print('False Negatives', FN)


# # Sensitivity

# In[66]:


TP=1039
FN=4250
sensitivity = (TP/(TP+FN))*100
print(sensitivity)


# # Specifity

# In[67]:


TN=39137
FP=785
specifity=(TN/(TN+FP))*100
specifity


# # Precision

# In[68]:


precision=(TP/(TP+FP))*100
precision


# # F_score

# In[69]:


F_score = (2*precision*sensitivity)/(precision+sensitivity)
F_score


# In[70]:


from sklearn.metrics import classification_report
print(classification_report(Y,y_pred))


# # ROC curve

# In[71]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[72]:


fpr,tpr,thresholds = roc_curve(Y,classifier.predict_proba(X)[:,1])
auc=roc_auc_score(Y,y_pred)
print(auc)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr, color='red', label = 'logit model (area=%0.2f)'%auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False positive rate or [1-True negative rate]')
plt.ylabel('Ture Positive Rate')


# In[73]:


data5['term_subscription'].value_counts()


# # In the above model the sensititvity of the model is very low owing to imbalance of the classes. Hence, we will down-sample the majority class 

# In[74]:


data5.head()


# In[75]:


# Separate majority and minority classes
data5_majority= data5[data5['term_subscription']==0]
data5_minority= data5[data5['term_subscription']==1]


# In[76]:


# Downsample majority class
from sklearn.utils import resample
data5_majority_downsampled = resample(data5_majority,
                                     replace = False, # sample without replacement
                                     n_samples=5289,  # to match minority class
                                     random_state=123) # reproducible results


# In[82]:


# Combine minority class with downsampled majority class
data5_downsampled = pd.concat([data5_majority_downsampled,data5_minority])


# In[84]:


# Display new class counts
data5_downsampled['term_subscription'].value_counts()


# In[85]:


data5_downsampled.head()


# # Applying logistic regression to above downsampled dataset

# In[86]:


# Dividing the data into Y and X units
Y=data5_downsampled.iloc[:,0]
X=data5_downsampled.iloc[:,1:]


# In[87]:


clf=LogisticRegression()
clf.fit(X,Y)


# In[88]:


clf.coef_


# In[89]:


clf.predict_proba(X)


# In[90]:


clf.predict(X)


# In[91]:


clf.score(X,Y)


# In[92]:


y_predictions=clf.predict(X)


# In[93]:


dataframe=pd.DataFrame({"Actual":Y,"Predicted":y_predictions})
dataframe


# In[94]:


confusion = confusion_matrix(Y,y_predictions)
print(confusion)


# In[99]:


TN=4175
TP=3978
FP=1114
FN=1311


# In[102]:


# Calculating Sensitivity
sensitivity1= TP/(TP+FN)*100
print(round(sensitivity1,2))


# In[103]:


# Calculating specificity
specificity1=TN/(TN+FP)*100
print(round(specificity1,2))


# In[104]:


# Calculating Precision
precision1=TP/(TP+FP)*100
print(round(precision1,2))


# In[106]:


# F_score
F_score1=(2*precision1*sensitivity1)/(precision1+sensitivity1)
F_score1


# In[107]:


# Classification report
print(classification_report(Y,y_predictions))


# In[109]:


# ROC curve

fpr,tpr,thresholds=roc_curve(Y,clf.predict_proba(X)[:,1])
auc=roc_auc_score(Y,y_predictions)
print(auc)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr, color='red', label = 'logit model (area=%0.2f)'%auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False positive rate or [1-True negative rate]')
plt.ylabel('Ture Positive Rate')


# In[ ]:




