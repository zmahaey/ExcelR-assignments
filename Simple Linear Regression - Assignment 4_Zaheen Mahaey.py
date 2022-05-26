#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[7]:


data = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 4_Linear regression\\delivery_time.csv")
data


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


data = data.rename({'Delivery Time': 'del_time', 'Sorting Time': 'sort_time'},axis=1)
data


# # Linear Regression model for delivery time

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


plt.figure(figsize=(5,5))
sns.scatterplot(x='sort_time', y='del_time',data= data)


# In[13]:


sns.distplot(data['del_time'])


# In[14]:


sns.distplot(data['sort_time'])


# In[15]:


import statsmodels.formula.api as smf


# In[16]:


model= smf.ols("del_time~sort_time", data = data).fit()


# In[17]:


sns.regplot(x='sort_time', y='del_time', data = data)


# In[18]:


model.params


# In[19]:


model.fittedvalues


# In[20]:


model.resid


# In[21]:


model.rsquared


# In[22]:


model.pvalues


# Since pvalues <0.05, we can say that the intercept and slope of sort_time is significant

# In[23]:


data['predicted_values']=model.fittedvalues
data['error_values']= model.resid
data


# In[24]:


mean_sq_values = data['error_values']**2
data['mean_sq_values']=mean_sq_values
data


# In[25]:


mean_sq_error = data['mean_sq_values'].mean()
mean_sq_error


# # Predicting new points

# In[26]:


new_data= pd.Series([2.5,5.5,7.5])
new_data


# In[27]:


data_predict = pd.DataFrame(new_data,columns=['sort_time'])
data_predict


# In[28]:


model.predict(data_predict)


# In[29]:


data_predict['del_time_new']=model.predict(data_predict)
data_predict


# In[30]:


model.predict(data)


# # Linear Regression model for salary hike

# In[1]:


import pandas as pd


# In[2]:


hike = pd.read_csv("C:\\Users\\zahee\\Desktop\\Data Science\\Assignments\\Assignment 4_Linear regression\\Salary_Data.csv")
hike


# In[3]:


hike.info()


# In[4]:


hike.isnull().sum()


# In[5]:


hike=hike.rename({'YearsExperience': 'yrs_exp','Salary': 'salary'}, axis=1)
hike


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


plt.figure(figsize=(5,5))
sns.scatterplot(x='yrs_exp', y='salary', data = hike)


# In[8]:


hike.corr()


# In[9]:


sns.distplot(hike['yrs_exp'])


# In[10]:


sns.distplot(hike['salary'])


# In[11]:


import statsmodels.formula.api as smf


# In[12]:


model=smf.ols("salary~yrs_exp",data=hike).fit()


# In[13]:


sns.regplot(x='yrs_exp',y='salary', data= hike)


# In[14]:


model.params


# In[15]:


model.fittedvalues


# In[16]:


model.resid


# In[17]:


model.rsquared


# In[18]:


model.pvalues


# Since pvalues<0.05, we can say that the intercept and slope for "yrs_exp" is significant

# In[19]:


hike['predicted_values']=model.fittedvalues
hike['error_values']=model.resid


# In[20]:


hike


# In[21]:


error_sq_values = hike['error_values']**2


# In[22]:


hike['error_sq_values']= error_sq_values
hike


# In[23]:


mean_sq_error = hike['error_sq_values'].mean()
mean_sq_error


# In[24]:


round(mean_sq_error,2)


# In[29]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line= 'q')
plt.title("Normal QQ Plot")
plt.show()


# In[41]:


def get_standardized_values(vals):
    return (vals - vals.mean())/vals.std()


# In[42]:


plt.scatter(get_standardized_values(model.fittedvalues),
           get_standardized_values(model.resid))

plt.title ('Residual PLot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# # Since there is non normality and unequal variance, applying log transformation

# In[30]:


import numpy as np
salary_hike=np.log(hike['salary'])
salary_hike


# In[31]:


hike['log_salary_hike']=salary_hike
hike


# In[32]:


plt.figure(figsize=(5,5))
sns.scatterplot(x='yrs_exp', y='log_salary_hike', data = hike)


# In[33]:


import numpy as np
model1=smf.ols("log_salary_hike~yrs_exp",data=hike).fit()


# In[34]:


sns.regplot(x="yrs_exp", y="log_salary_hike", data=hike)


# In[35]:


model1.rsquared


# In[36]:


model1.params,model1.pvalues


# In[37]:


model1.fittedvalues


# In[38]:


model1.resid


# In[39]:


mean_sq_error_1= ((model1.resid)**2).mean()
mean_sq_error_1


# In[40]:


import statsmodels.api as sm
qqplot=sm.qqplot(model1.resid,line= 'q')
plt.title("Normal QQ Plot")
plt.show()


# In[43]:


plt.scatter(get_standardized_values(model1.fittedvalues),
           get_standardized_values(model1.resid))

plt.title ('Residual PLot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# # Since the normality, variance across the model has improved, we go ahead to build the model

# # Predicting new values for salary

# In[87]:


new_data=pd.Series([3,7,10])
new_data


# In[88]:


new_data_predicted = pd.DataFrame(new_data, columns = ['yrs_exp'])
new_data_predicted


# In[90]:


# Calculating log values for salary
log_predicted_data=model1.predict(new_data_predicted)
log_predicted_data


# In[92]:


new_data_predicted['log_predicted_data']= log_predicted_data
new_data_predicted


# In[99]:


new_data_predicted['salary_model1']=np.exp(new_data_predicted['log_predicted_data'])
new_data_predicted


# In[101]:


Salary_Prediction_final = np.exp(model1.predict(hike))


# In[102]:


Salary_Prediction_final


# In[ ]:




