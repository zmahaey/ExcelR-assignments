#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Loading the dataset
df=pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 15_Random Forests\\Fraud_check.csv")
df


# In[3]:


df['Taxable.Income'].max()


# In[4]:


# Creating a column 'income' which has taxable_income <= 30000 as "Risky" and others as "Good"
df['income']=pd.cut(x=df['Taxable.Income'], bins=[0,30000,100000], right=True, labels=['risky','good'])
df


# In[5]:


df=df.drop('Taxable.Income',axis=1)
df.head()


# In[6]:


# Converting categorical values to numerical values through label encoder
labelencoder=LabelEncoder()
df.iloc[:,0]=labelencoder.fit_transform(df.iloc[:,0])
df.iloc[:,1]=labelencoder.fit_transform(df.iloc[:,1])
df.iloc[:,4]=labelencoder.fit_transform(df.iloc[:,4])
df.iloc[:,5]=labelencoder.fit_transform(df.iloc[:,5])


# In[7]:


df.head()


# In[8]:


# Visualizing the data
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df, hue='income',palette='tab10')
plt.show()


# In[126]:


X=df.iloc[:,0:5]
Y=df.iloc[:,5]


# In[127]:


from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
scaled_data=scale.fit_transform(X)
scaled_data


# In[128]:


x=scaled_data


# # Applying Random Forest algorithm by splitting into training & test data and calculating the accuracy

# In[129]:


from sklearn.ensemble import RandomForestClassifier


# #### stratify ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y”

# In[130]:


x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.5, random_state=7, stratify=Y)


# In[131]:


model= RandomForestClassifier(n_estimators=1000, max_features=3)
model.fit(x_train,y_train)


# In[132]:


preds=model.predict(x_test)
preds


# In[133]:


from sklearn.metrics import confusion_matrix


# In[134]:


cm=confusion_matrix(y_test, preds)
print(cm)


# # Since the False Negatives are quite high in number due to class imbalance, trying the K-Fold cross validation technique

# In[135]:


'''Stratified k-fold cross-validation is the same as just k-fold cross-validation, But Stratified k-fold cross-
validation, it does stratified sampling instead of random sampling.'''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[136]:


kfold=StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
model1= RandomForestClassifier(n_estimators=1000, max_features=3)
result = cross_val_score(model1,x,Y,cv=kfold)
print(result.mean())


# ### Thus, we can see that we get an accuracy of 74.3% with stratified K-Fold cross validation technique through random forest

# # Checking for Random Forest accuracy through upsampling

# In[78]:


df.head()


# In[80]:


df['income'].value_counts()


# In[81]:


# Separate into majority and minority classes
df_majority = df[df['income']==0]
df_minority = df[df['income']==1]


# In[115]:


# Upsample minority class
from sklearn.utils import resample
df_minority_upsample = resample(df_minority,
                                  replace=True,
                                 n_samples=476,
                                 random_state=7)


# In[116]:


# Combine upsampled minority class and majority class
df_upsampled= pd.concat([df_minority_upsample,df_majority])
df_upsampled


# In[104]:


X=df_downsampled.iloc[:,0:5]
Y=df_downsampled.iloc[:,5]


# In[105]:


# Dividing the data into train and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=7)


# In[106]:


model3= RandomForestClassifier(n_estimators=1000, max_features=3)
model3.fit(x_train,y_train)


# In[107]:


preds1=model3.predict(x_test)
preds1


# In[108]:


cm=confusion_matrix(y_test,preds1)
print(cm)


# In[109]:


TN=18
TP=8
FN=13
FP=11


# In[110]:


# Sensitivity
Sensitivity = TP/(TP+FN)
print(Sensitivity*100)


# In[111]:


# Specificity
Specificity = TN/(TN+FP)
print(Specificity*100)


# In[112]:


# Precision
Precision = TP/(TP+FP)
print(Precision*100)


# In[113]:


# F Score
F_score=(2*Precision*Sensitivity)/(Precision+Sensitivity)
print(F_score*100)


# In[114]:


from sklearn.metrics import classification_report
print(classification_report(y_test,preds1))


# # Thus, we can see that out of the three approaches adopted above, Stratified K-Fold works the best for Random Forest with the accuracy of 74.3% 

# In[ ]:




