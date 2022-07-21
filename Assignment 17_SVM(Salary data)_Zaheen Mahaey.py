#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Loading training dataset
data =pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 17_SVM\\SalaryData_Train.csv")
data


# In[3]:


# Checking duplicates
data[data.duplicated()]


# In[4]:


df=data.drop_duplicates()
df


# In[5]:


df.info()


# In[6]:


# Resetting row index
df=df.reset_index()
df.head()


# In[7]:


df=df.drop("index",axis=1)
df


# In[8]:


# Applying labelencoder to train data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df.iloc[:,1]=labelencoder.fit_transform(df.iloc[:,1])
df.iloc[:,2]=labelencoder.fit_transform(df.iloc[:,2])
df.iloc[:,4]=labelencoder.fit_transform(df.iloc[:,4])
df.iloc[:,5]=labelencoder.fit_transform(df.iloc[:,5])
df.iloc[:,6]=labelencoder.fit_transform(df.iloc[:,6])
df.iloc[:,7]=labelencoder.fit_transform(df.iloc[:,7])
df.iloc[:,8]=labelencoder.fit_transform(df.iloc[:,8])
df.iloc[:,12]=labelencoder.fit_transform(df.iloc[:,12])
df.iloc[:,13]=labelencoder.fit_transform(df.iloc[:,13])


# In[9]:


df


# In[10]:


# Calculating PPS score
import ppscore as pps
colnames=list(df.columns)
for i in colnames:
    z=pps.score(df,i,'Salary')
    print("The PPS score for {} with Salary is {}".format(i,z["ppscore"]))


# ### Since no variable contributes to predicting the salary, we will retain all features to train the model

# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(df['Salary'])
plt.show()


# In[12]:


# Separating numerical variables from training dataset
df_num = df[['age','capitalgain','capitalloss','hoursperweek']]
df_num


# In[13]:


# Normalizing the numerical features of training data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaled = scaler.fit_transform(df_num)
scaled


# In[14]:


scaled_df_num=pd.DataFrame(scaled,columns=['age','capitalgain','capitalloss','hoursperweek'])
scaled_df_num


# In[15]:


# Separating categorical variables from the training data 
df_cat= df.drop(['age','capitalgain','capitalloss','hoursperweek'],axis=1)
df_cat


# In[16]:


# Joining the numerical and categorical features to form the training data to be fed to the model
df_train=pd.concat([scaled_df_num,df_cat],axis=1)
df_train


# In[17]:


# Load test dataset
file_data=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 17_SVM\\SalaryData_Test.csv")
file_data


# In[18]:


# Checking duplicates in test data
file_data[file_data.duplicated()]


# In[19]:


file=file_data.drop_duplicates()
file


# In[20]:


# Reset row index in test data
file=file.reset_index()
file


# In[21]:


file=file.drop('index',axis=1)
file


# In[22]:


file.info()


# In[23]:


# Applying labelencoder to test data
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


# In[24]:


file


# In[25]:


# Separating numerical features in test data
file_num=file[['age','capitalgain','capitalloss','hoursperweek']]
file_num


# In[26]:


# Normalizing the numerical features in test data
scaled2 = scaler.fit_transform(file_num)
scaled2


# In[27]:


scaled_file_num=pd.DataFrame(scaled2,columns=['age','capitalgain','capitalloss','hoursperweek'])
scaled_file_num


# In[28]:


# Separating the categorical features in the test data
file_cat=file.drop(['age','capitalgain','capitalloss','hoursperweek'],axis=1)
file_cat


# In[29]:


# Joining the test and numerical features to form the test data to be tested
file_test=pd.concat([scaled_file_num,file_cat],axis=1)
file_test


# In[30]:


#For training dataset
X=df_train.iloc[:,:-1]
Y=df_train.iloc[:,-1]


# In[47]:


# For Test dataset
x=file_test.iloc[:,:-1]
y=file_test.iloc[:,-1]


# # GridSearchCV

# In[32]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[34]:


clf=SVC()
param_grid=[{'kernel':['rbf'],'gamma':[100,50,10,1,0.5,0.1,0.01,0.001,0.0001],
            'C': [15,14,13,12,11,10,5,1,0.5,0.1,0.01,0.001,0.0001]}]
gsv=GridSearchCV(clf,param_grid,cv=2)
gsv.fit(X,Y)


# In[36]:


gsv.best_params_,gsv.best_score_


# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[38]:


clf=SVC(gamma=0.01 , C=12 )
clf.fit(X,Y)
y_preds_train=clf.predict(X)
acc=accuracy_score(Y,y_preds_train)
print("accuracy=",acc)
print(confusion_matrix(Y,y_preds_train))


# In[48]:


y_preds_test=clf.predict(x)
acc=accuracy_score(y,y_preds_test)
print("accuracy=",acc)
print(confusion_matrix(y,y_preds_test))


# In[49]:


TN=9927
TP=1954
FN=1556
FP=693


# In[50]:


# Sensitivity
sensitivity = TP/(TP+FN)
print(sensitivity*100)


# In[51]:


# Specificity
specificity = TN/(TN+FP)
print(specificity*100)


# In[53]:


# Precision
precision= TP/(TP+FP)
print(precision*100)


# In[54]:


# F-score
F_score=(2*precision*sensitivity)/(precision+sensitivity)
print(F_score)


# In[57]:


from sklearn.metrics import classification_report
print(classification_report(y,y_preds_test))


# In[59]:


# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr,tpr,thresholds = roc_curve(y,y_preds_test)
auc=roc_auc_score(y,y_preds_test)
print(auc)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr, color='red', label = 'logit model (area=%0.2f)'%auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False positive rate or [1-True negative rate]')
plt.ylabel('Ture Positive Rate')


# In[ ]:




